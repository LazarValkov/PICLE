import argparse
import torch
torch.use_deterministic_algorithms(True)
from Experiments.Interface.Experiment import Experiment
from Experiments.BELL.NNTrainer import BELLNNTrainer
from Experiments.BELL.ProblemSequenceGenerators import BELLProblem_SequencesGenerator
from CL.Baselines.Standalone import Standalone
from CL.Baselines.MNTDP import MNTDP
from CL.Baselines.MultiheadBaselines import EWC, ERwRingBuffer
from CL.Baselines.HOUDINI import HOUDINI
from CL.PICLE.PICLE import PICLE
from CL.Baselines.Random import ModularCLAlg_RandomSearch


class BELLExperiment(Experiment):
    def get_nn_trainer(self, problem):
        return BELLNNTrainer(problem, self.device)


# Data settings
num_tr_dp_max_perf = (30000, None, None)
num_tr_dp_lower_perf = (10000, 100, None)
num_tr_dp_lower_perf_hlt = (10000, None, 15)
num_tr_dp_lower_perf_few = (10, None, None)

num_val = num_test = 5000

# Optimiser Settings
batch_size = 32
num_epochs = 1200

# lml algorithm settings
enable_finetuning = False
if_finetuning_finetune_a_copy = True

def get_cl_alg(alg_name, random_seed, device, tmp_folder):
    # PICLE parameters
    pt_hyperparam_soft_type_target_dim = 20
    pt_hyperparam_softmax_temp = 0.001
    nt_hyperparam_num_input_points_stored_per_module = 40
    nt_hyperparam_lmin = 3

    # Hv2_PT, Hv2_NT, Hv2, MNTDP, random, HOUDINI, standalone, ewc
    if alg_name == "PICLE_PT":
        return PICLE(pt_hyperparam_soft_type_target_dim, pt_hyperparam_softmax_temp,
                     nt_hyperparam_num_input_points_stored_per_module, nt_hyperparam_lmin,
                     random_seed, device, num_epochs, f"{tmp_folder}/PICLE_PT/lib",
                     enable_finetuning, if_finetuning_finetune_a_copy,
                     evaluations_cache_root_folder=f"{tmp_folder}/PICLE_PT/eval_cache",
                     _debug_search_perceptual_transfer=True,
                     _debug_search_latent_transfer=False
                     )
    if alg_name == "PICLE_NT":
        return PICLE(pt_hyperparam_soft_type_target_dim, pt_hyperparam_softmax_temp,
                     nt_hyperparam_num_input_points_stored_per_module, nt_hyperparam_lmin,
                     random_seed, device, num_epochs, f"{tmp_folder}/PICLE_NT/lib",
                     enable_finetuning, if_finetuning_finetune_a_copy,
                     evaluations_cache_root_folder=f"{tmp_folder}/PICLE_NT/eval_cache",
                     _debug_search_perceptual_transfer=False,
                     _debug_search_latent_transfer=True
                     )
    if alg_name == "PICLE":
        return PICLE(pt_hyperparam_soft_type_target_dim, pt_hyperparam_softmax_temp,
                     nt_hyperparam_num_input_points_stored_per_module, nt_hyperparam_lmin,
                     random_seed, device, num_epochs, f"{tmp_folder}/PICLE/lib",
                     enable_finetuning, if_finetuning_finetune_a_copy,
                     evaluations_cache_root_folder=f"{tmp_folder}/PICLE/eval_cache",
                     _debug_search_perceptual_transfer=True,
                     _debug_search_latent_transfer=True
                     )
    if alg_name == "MNTDP":
        return MNTDP(random_seed, device, num_epochs, f"{tmp_folder}/MNTDP/lib",
                     enable_finetuning, if_finetuning_finetune_a_copy,
                     evaluations_cache_root_folder=f"{tmp_folder}/MNTDP/eval_cache")

    if alg_name == "random":
        return ModularCLAlg_RandomSearch(random_seed, device, num_epochs, f"{tmp_folder}/RS/lib",
                                         enable_finetuning, if_finetuning_finetune_a_copy,
                                         evaluations_cache_root_folder=f"{tmp_folder}/RS/eval_cache")
    if alg_name == "HOUDINI":
        return HOUDINI(random_seed, device, num_epochs, f"{tmp_folder}/Exh/lib",
                       enable_finetuning, if_finetuning_finetune_a_copy,
                       evaluations_cache_root_folder=f"{tmp_folder}/Exh/eval_cache")
    if alg_name == "standalone":
        return Standalone(random_seed, device, num_epochs)
    if alg_name == "ewc":
        return EWC(random_seed, device, num_epochs, f"{tmp_folder}/EWC/lib")
    if alg_name == "ER":
        return ERwRingBuffer(random_seed, device, num_epochs, f"{tmp_folder}/ER/lib",
                             num_examples_to_store_per_class=nt_hyperparam_num_input_points_stored_per_module//2)

    raise ValueError("Invalid value for alg_name provided!")


def get_cl_alg_no_cache(alg_name, random_seed, device, tmp_folder):
    # hv2 parameters
    pt_hyperparam_soft_type_target_dim = 20
    pt_hyperparam_softmax_temp = 0.001
    nt_hyperparam_num_input_points_stored_per_module = 40
    nt_hyperparam_lmin = 3

    #soft_type_target_dim = 20
    #num_inputs_stored_per_module = 40
    if alg_name == "PICLE_PT":
        return PICLE(pt_hyperparam_soft_type_target_dim, pt_hyperparam_softmax_temp,
                     nt_hyperparam_num_input_points_stored_per_module, nt_hyperparam_lmin,
                     random_seed, device, num_epochs, f"{tmp_folder}/PICLE_PT/lib",
                     enable_finetuning, if_finetuning_finetune_a_copy,
                     evaluations_cache_root_folder=None,
                     _debug_search_perceptual_transfer=True,
                     _debug_search_latent_transfer=False
                     )
    if alg_name == "PICLE":
        return PICLE(pt_hyperparam_soft_type_target_dim, pt_hyperparam_softmax_temp,
                     nt_hyperparam_num_input_points_stored_per_module, nt_hyperparam_lmin,
                     random_seed, device, num_epochs, f"{tmp_folder}/PICLE/lib",
                     enable_finetuning, if_finetuning_finetune_a_copy,
                     evaluations_cache_root_folder=None,
                     _debug_search_perceptual_transfer=True,
                     _debug_search_latent_transfer=True
                     )
    if alg_name == "MNTDP":
        return MNTDP(random_seed, device, num_epochs, f"{tmp_folder}/MNTDP/lib",
                     enable_finetuning, if_finetuning_finetune_a_copy,
                     evaluations_cache_root_folder=None)

    raise ValueError("Invalid value for alg_name provided!")


experiments_dict = {}

""""""
experiments_dict["S_out_star_star"] = {
    "seq_folder_name": "S_out_star_star",
    "seq_gen_fn_name": "get_S_out_star_star",
    "length": 6,
    "random_seeds": [20, 21, 22] # 23, 24,
}

experiments_dict["S_minus"] = {
    "seq_folder_name": "S_minus",
    "seq_gen_fn_name": "get_S_minus",
    "length": 6,
    "random_seeds": [25, 26, 27] # 28, 29
}

experiments_dict["S_pl"] = {
    "seq_folder_name": "S_pl",
    "seq_gen_fn_name": "get_S_pl",
    "length": 6,
    "random_seeds": [30, 31, 32] # 33, 34
}

experiments_dict["S_plus"] = {
    "seq_folder_name": "S_plus",
    "seq_gen_fn_name": "get_S_plus",
    "length": 6,
    "random_seeds": [35, 36, 37] # 38, 39
}

experiments_dict["S_out"] = {
    "seq_folder_name": "S_out",
    "seq_gen_fn_name": "get_S_out",
    "length": 6,
    "random_seeds": [40, 41, 42] # , 43, 44
}

experiments_dict["S_out_star"] = {
    "seq_folder_name": "S_out_star",
    "seq_gen_fn_name": "get_S_out_star",
    "length": 6,
    "random_seeds": [45, 46, 47] # 48, 49
}

experiments_dict["S_few"] = {
    "seq_folder_name": "S_few",
    "seq_gen_fn_name": "get_S_few",
    "length": 6,
    "random_seeds": [50, 51, 52] # 53, 54
}

experiments_dict["S_in"] = {
    "seq_folder_name": "S_in",
    "seq_gen_fn_name": "get_S_in",
    "length": 6,
    # "random_seeds": [55, 56, 57] # 58, 59
    "random_seeds": [55, 56, 54]
}

experiments_dict["S_sp"] = {
    "seq_folder_name": "S_sp",
    "seq_gen_fn_name": "get_S_sp",
    "length": 6,
    # "random_seeds": [55, 56, 57] # 58, 59
    "random_seeds": [55, 56, 54]
}

"""
experiments_dict["S_long"] = {
    "seq_folder_name": "S_long",
    "seq_gen_fn_name": "get_S_long",
    "length": 100,
    "random_seeds": [60]
}
"""


def evaluate_seed(experiment_name, random_seed, seq_folder_name, seq_gen_fn_name, length,
                 alg_name, device, max_time, start_from_idx=0):
    tmp_folder = f"results/BELL/{seq_folder_name}/rs_{random_seed}"
    results_folder = f"{tmp_folder}/Results"

    c_seq_generator = BELLProblem_SequencesGenerator(batch_size, num_tr_dp_max_perf,
                                                     num_tr_dp_lower_perf, num_tr_dp_lower_perf_hlt, num_tr_dp_lower_perf_few,
                                                     num_val, num_test)
    c_seq = getattr(c_seq_generator, seq_gen_fn_name)(random_seed, length=length)

    print("========================================================")
    print(f"EVALUATING {experiment_name} w random_seed = {random_seed}")
    for p in c_seq:
        print(p.num_id, p.name, p.total_num_tr_datapoints)

    if experiment_name == "S_long":
        alg = get_cl_alg_no_cache(alg_name, random_seed, device, tmp_folder)
    else:
        alg = get_cl_alg(alg_name, random_seed, device, tmp_folder)
    experiment = BELLExperiment(max_time, device, alg, c_seq, results_folder)

    if experiment_name == "S_long":
        print("Setting max num programs.")
        experiment.run_sequence(start_from_idx=start_from_idx, max_num_programs=(1+8+2*8))  # 1 + 3L
    else:
        experiment.run_sequence(start_from_idx=start_from_idx)


def run_experiment(experiment_name, alg_name, device, max_time, start_from_idx=0, _dbg=False):
    c_exp_dict = experiments_dict[experiment_name]

    for c_random_seed in c_exp_dict["random_seeds"]:
        evaluate_seed(experiment_name, c_random_seed, c_exp_dict["seq_folder_name"],
                      c_exp_dict["seq_gen_fn_name"], c_exp_dict["length"],
                      alg_name, device, max_time, start_from_idx=start_from_idx)
        if _dbg:  # Run for only one random seed in debug mode
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sequence", help=f"One of: {list(experiments_dict.keys())}", type=str)
    parser.add_argument("-cl_alg", help=f"One of: PICLE, MNTDP, random, HOUDINI, standalone, ewc, PICLE_PT, PICLE_NT", type=str)
    parser.add_argument("-device", help=f"the device pytorch will train+evaluate the NNs on", type=str)
    parser.add_argument("-dbg", help=f"If true, train each NN for only 1 epoch", action='store_true')
    args = parser.parse_args()
    print(f"-sequence = {args.sequence}")
    print(f"-cl_alg = {args.cl_alg}")
    print(f"-device = {args.device}")
    print(f"-dbg = {args.dbg}")

    if args.dbg:
        num_epochs = 1

    run_experiment(args.sequence, args.cl_alg, args.device, max_time=None, _dbg=args.dbg)
