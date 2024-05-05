from CL.Baselines.HOUDINI import *


class ModularCLAlg_RandomSearch(HOUDINI):
    def get_name(self):
        return "ModularRS"

    def _get_all_module_combinations(self, problem: Problem, random_obj: Random):
        modules_names_list_of_lists = self._get_modules_names_list_of_lists(problem)
        for mnl in modules_names_list_of_lists:
            mnl.insert(0, None)
        all_module_combinations = list(itertools.product(*modules_names_list_of_lists))

        standalone_prog = (None,) * problem.architecture_class.get_num_modules()

        all_module_combinations.remove(standalone_prog)
        random_obj.shuffle(all_module_combinations)
        all_module_combinations.insert(0, standalone_prog)
        return all_module_combinations

    def get_should_limit_evaluations_by_time(self):
        return True

    def get_should_limit_evaluations_by_number_of_paths(self):
        return False