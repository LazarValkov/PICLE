import copy
import numpy as np
import scipy.special

from CL.Interface.NNModule import NNModule
from CL.Interface.Problem import Problem
from CL.PICLE.Library import PICLELibrary
from typing import List, Tuple, Dict, Union
from CL.PICLE.SoftType import SoftType


class PICLE_PTSearch():
    """
    Search for the best low-level transfer program.
    """
    def __init__(self, problem: Problem,
                 lib: PICLELibrary,
                 soft_type_target_dim: int,
                 device: str,
                 standalone_prog_performance: float,
                 verbose: bool = False,
                 prior_softmax_temp=1.,
                 ):
        self.problem = problem
        self.lib = lib
        self.soft_type_target_dim = soft_type_target_dim
        self.device = device
        self.verbose = verbose

        self.data_loader_tr = problem.get_tr_data_loader()

        self.architecture_class = problem.architecture_class
        self.num_module_layers = self.architecture_class.get_num_modules()

        self.evaluated_partial_programs : List[Tuple[Union[Tuple[str], Tuple], float]] = [
            ((), standalone_prog_performance)
        ]

        self.partial_program_to_log_prior = {(): 0.}
        self.partial_program_to_log_posterior = {}  # log p(Partial_program | x)
        self.partial_prog_to_dict_new_module_to_log_likelihood = {}  # log (h | NewModule, Partial_program)

        # this maps a partial program to the lib items of the modules than can be used to expand it upwards
        self.dict_partial_prog_to_list_lib_items_for_expansion = {}
        self.dict_partial_program_to_log_marginal_of_last_hidden_state = {}

        # cache the last suggestion, so if it's not evaluated, we can re-suggest it.
        self.last_suggestion_partial_program_strs = None
        self.last_suggestion_aq_value = None
        self.last_suggestion_test_program_idx = None

        self.intial_programs_iterated = False
        self.best_st_pt_programs_iterator = None

        self.no_more_suggestions = False

        # calculate the performance-proportional probabilities of each module for each layer
        self.module_type_to_sum_performances = {}
        for c_module_type, c_layer_modules in self.lib.items_by_module_type.items():
            c_all_performances = []
            for c_lib_item in c_layer_modules:
                c_all_performances += c_lib_item.performances
            self.module_type_to_sum_performances[c_module_type] = sum(c_all_performances)

        self.softmax_temperature = prior_softmax_temp  #  0.001 # 0.0002  # 1.
        self.module_type_to_log_sum_exp_performances = {}
        for c_module_type, c_layer_modules in self.lib.items_by_module_type.items():
            c_all_performances = []
            for c_lib_item in c_layer_modules:
                c_all_performances += c_lib_item.performances

            c_perf_temp_applied = np.array(c_all_performances) / self.softmax_temperature
            self.module_type_to_log_sum_exp_performances[c_module_type] = scipy.special.logsumexp(c_perf_temp_applied)

    def _get_pp_outputs(self, partial_program_strs: Tuple[str]):
        if partial_program_strs == ():
            # get the soft type of the input
            dataset_tr_np = self.data_loader_tr.dataset.tensors[0].numpy()
            return dataset_tr_np
        else:
            partial_program_modules: Tuple[NNModule] = tuple(self.lib[m_str].module for m_str in partial_program_strs)
            pp_outputs_np = self.architecture_class.get_partial_program_outputs_PT(partial_program_modules,
                                                                                   self.data_loader_tr, self.device)
            c_output_shape = self.architecture_class.get_layers_input_shape(len(partial_program_strs))
            if type(pp_outputs_np) == list and type(c_output_shape) == tuple:
                # we need to flatten the list
                assert len(pp_outputs_np) * pp_outputs_np[0].shape[1] == c_output_shape[0]
                pp_outputs_np = np.concatenate(pp_outputs_np, axis=1)

            return pp_outputs_np

    def _get_model_log_posterior(self, c_partial_program_strs):
        assert None not in c_partial_program_strs
        assert len(c_partial_program_strs) > 0

        if c_partial_program_strs in self.partial_program_to_log_posterior.keys():
            return self.partial_program_to_log_posterior[c_partial_program_strs]

        log_posterior = 0.
        log_prior = self.partial_program_to_log_prior[c_partial_program_strs]
        log_posterior += log_prior

        # log_input_given_layer_and_output = []
        # log p(z | h, theta_i), where h = theta_i(z), i  = l-1
        for l in range(1, len(c_partial_program_strs) + 1):
            c_sub_pp_strs = c_partial_program_strs[:l]

            c_log = 0

            # log p(z | theta_i)
            c_last_module = c_sub_pp_strs[-1]
            c_ll_of_last_hidden_state = self.partial_prog_to_dict_new_module_to_log_likelihood[c_sub_pp_strs[:-1]][c_last_module]

            c_log += c_ll_of_last_hidden_state

            if l < len(c_partial_program_strs):
                c_log_marginal_of_last_hidden_state = self.dict_partial_program_to_log_marginal_of_last_hidden_state[c_sub_pp_strs[:-1]]
                c_log -= c_log_marginal_of_last_hidden_state # np.log(p_h_thetha)

            log_posterior += c_log

        self.partial_program_to_log_posterior[c_partial_program_strs] = log_posterior
        return log_posterior

    def get_module_log_prior(self, m_lib_item, soft_type_idx):
        # this version uses softmax
        c_performance = m_lib_item.performances[soft_type_idx]
        c_perf_w_temp = c_performance / self.softmax_temperature

        c_log_prob = c_perf_w_temp - self.module_type_to_log_sum_exp_performances[m_lib_item.module_type]
        return c_log_prob

    def _expand_partial_program_upwards(self, c_partial_program_strs: Union[Tuple, Tuple[str]]):
        # c_partial_program is the "tallest" number of modules explored so far.
        assert c_partial_program_strs not in self.dict_partial_prog_to_list_lib_items_for_expansion.keys()
        assert c_partial_program_strs not in self.partial_prog_to_dict_new_module_to_log_likelihood

        next_layer_index = len(c_partial_program_strs)
        next_module_type = self.problem.architecture_class.get_module_types()[next_layer_index]

        if next_module_type not in self.lib.items_by_module_type.keys():
            return None

        choices_for_next_module_lib_items = copy.copy(self.lib.items_by_module_type[next_module_type])

        if len(choices_for_next_module_lib_items) == 0:
            return None

        # ***** find the best new addition *****

        # For each, compute P(Model | x) and then select the highest one
        c_pp_outputs = self._get_pp_outputs(c_partial_program_strs)
        c_pp_outputs_processed = SoftType.process_data_points(c_pp_outputs)

        # I need 2 dictionaries:
        # - Containing [pprogram -> log_P(pprogram | x)]
        # - Containing [prev_pprogram, new_module] -> log_P(z | new_module), where z = prev_pprogram(x)

        # c_dict_pp_and_module_to_module_log_likelihood = {}   # P(z | new_module)
        c_dict_module_id_to_log_likelihood = {}
        for m_lib_item in choices_for_next_module_lib_items:
            c_lls = [-c_input_st.get_nll_of_processed_datapoints(c_pp_outputs_processed) for c_input_st in
                      m_lib_item.input_soft_types]
            c_max_ll = max(c_lls)
            c_max_ll_idx = c_lls.index(c_max_ll)

            # use the distribution with maximum likelihood
            c_dict_module_id_to_log_likelihood[m_lib_item.name] = c_max_ll

            c_partial_program_log_prior = self.partial_program_to_log_prior[c_partial_program_strs]
            c_new_module_log_prior = self.get_module_log_prior(m_lib_item, c_max_ll_idx)
            c_new_partial_program_strs = c_partial_program_strs + (m_lib_item.name,)
            c_new_partial_program_log_prior = c_partial_program_log_prior + c_new_module_log_prior
            self.partial_program_to_log_prior[c_new_partial_program_strs] = c_new_partial_program_log_prior

            # compute the prior of the resulting partial program

        # h = c_pp_outputs
        # p(h) = sum_i p(h | M_new) p(M_new)
        # where M_new is different for the different soft types as well
        # c_log_marginal_of_last_hidden_state = 0.   #
        c_log_marginal_of_last_hidden_state_log_summands = []

        for m_lib_item in choices_for_next_module_lib_items:
            for c_st_idx, c_input_st in enumerate(m_lib_item.input_soft_types):
                c_ll = -c_input_st.get_nll_of_processed_datapoints(c_pp_outputs_processed)
                c_log_prior = self.get_module_log_prior(m_lib_item, c_st_idx)
                c_log_marginal_of_last_hidden_state_log_summands.append(c_ll + c_log_prior)
        c_log_marginal_of_last_hidden_state = scipy.special.logsumexp(c_log_marginal_of_last_hidden_state_log_summands)
        self.dict_partial_program_to_log_marginal_of_last_hidden_state[c_partial_program_strs] = c_log_marginal_of_last_hidden_state

        ######################################


        self.partial_prog_to_dict_new_module_to_log_likelihood[c_partial_program_strs] = c_dict_module_id_to_log_likelihood

        c_dict_new_module_to_model_log_posterior = {}
        for m_lib_item in choices_for_next_module_lib_items:
            c_new_partial_program_strs = c_partial_program_strs + (m_lib_item.name,)
            c_dict_new_module_to_model_log_posterior[m_lib_item.name] = self._get_model_log_posterior(c_new_partial_program_strs)

        # - order the lib items by log posterior
        choices_for_next_module_lib_items_ordered = sorted(choices_for_next_module_lib_items,
                                                           key=lambda m_lib_item: c_dict_new_module_to_model_log_posterior[m_lib_item.name],
                                                           reverse=True)

        # add the closest module choice in terms of soft types to the partial program
        selected_next_module_lib_item = choices_for_next_module_lib_items_ordered.pop(0)
        new_partial_program_strs = c_partial_program_strs + (selected_next_module_lib_item.name,)

        # cache the calculations
        self.dict_partial_prog_to_list_lib_items_for_expansion[c_partial_program_strs] = choices_for_next_module_lib_items_ordered
        # self.dict_partial_prog_to_dict_of_module_ids_to_nll[c_partial_program_strs] = c_dict_module_id_to_st_dist

        return new_partial_program_strs

    def _iterate_best_PT_programs(self):
        c_partial_program_strs = ()

        while len(c_partial_program_strs) < self.num_module_layers:  # update the condition later
            # add the closest module choice in terms of soft types to the partial program
            c_partial_program_strs = self._expand_partial_program_upwards(c_partial_program_strs)
            if c_partial_program_strs is None:
                break
            yield c_partial_program_strs

    def suggest_next_program(self):
        # if the last computed suggestion was not used
        if self.last_suggestion_partial_program_strs is not None:
            if self.verbose:
                print(f"Suggesting program = {self.last_suggestion_partial_program_strs}")
            return self.last_suggestion_partial_program_strs, self.last_suggestion_aq_value

        if self.no_more_suggestions:
            return None

        # First, check if we need to iterate the initial programs first
        if not self.intial_programs_iterated:
            if self.best_st_pt_programs_iterator is None:
                self.best_st_pt_programs_iterator = iter(self._iterate_best_PT_programs())
            try:
                self.last_suggestion_partial_program_strs = next(self.best_st_pt_programs_iterator)
                self.last_suggestion_aq_value = float("-inf")
                self.last_suggestion_test_program_idx = -1
                return self.last_suggestion_partial_program_strs, self.last_suggestion_aq_value
            except StopIteration:
                self.intial_programs_iterated = True

        self.no_more_suggestions = True
        return None

    def evaluate_suggested_program(self, eval_function):
        assert self.last_suggestion_partial_program_strs is not None

        if self.verbose:
            print("------------------------------------")
            print(f"PT, evaluating: {self.last_suggestion_partial_program_strs}, aq_value = {self.last_suggestion_aq_value}")

        # evaluate the suggested program
        suggested_partial_program_strs = self.last_suggestion_partial_program_strs
        suggested_program_strs = suggested_partial_program_strs + (None,) * (self.num_module_layers - len(suggested_partial_program_strs))

        c_eval_result = eval_function(suggested_program_strs)
        val_loss = c_eval_result.val_loss

        if not self.intial_programs_iterated:
            self.evaluated_partial_programs.append((suggested_partial_program_strs, val_loss))

            # reset the last partial program suggestion state
            self.last_suggestion_partial_program_strs = None
            self.last_suggestion_aq_value = None
            self.last_suggestion_test_program_idx = None

            return c_eval_result



