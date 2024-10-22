import abc
import os
import glob
import random
import torch

from typing import List, NamedTuple, Type
from libero.libero import get_libero_path
from libero.libero.benchmark.libero_suite_task_map import libero_task_map
from libero.libero.benchmark.yy_suite_task_map import yy_task_map
import json

BENCHMARK_MAPPING = {}


def register_benchmark(target_class):
    """We design the mapping to be case-INsensitive."""
    BENCHMARK_MAPPING[target_class.__name__.lower()] = target_class


def get_benchmark_dict(help=False):
    if help:
        print("Available benchmarks:")
        for benchmark_name in BENCHMARK_MAPPING.keys():
            print(f"\t{benchmark_name}")
    return BENCHMARK_MAPPING


def get_benchmark(benchmark_name):
    return BENCHMARK_MAPPING[benchmark_name.lower()]


def print_benchmark():
    print(BENCHMARK_MAPPING)


def create_reverse_mapping(mapping):
    """Create a reverse mapping from values to keys."""
    reverse_mapping = {}
    for key, values in mapping.items():
        for value in values:
            reverse_mapping.setdefault(value, []).append(key)
    return reverse_mapping

def find_keys_by_value(mapping, target_value):
    """Find all keys associated with a given value in the mapping."""
    reverse_mapping = create_reverse_mapping(mapping)
    return reverse_mapping.get(target_value, [])

class Task(NamedTuple):
    name: str
    language: str
    problem: str
    problem_folder: str
    bddl_file: str
    init_states_file: str


def grab_language_from_filename(x, is_yy=False):
    if is_yy:
        language = " ".join(x.split("SCENE")[-1][1:].split("_"))
    elif x[0].isupper():  # LIBERO-100
        if "SCENE10" in x:
            language = " ".join(x[x.find("SCENE") + 8 :].split("_"))
        else:
            language = " ".join(x[x.find("SCENE") + 7 :].split("_"))
    else:
        language = " ".join(x.split("_"))
    en = language.find(".bddl")
    return language[:en]


libero_suites = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
]
task_maps = {}
max_len = 0
for libero_suite in libero_suites:
    task_maps[libero_suite] = {}

    for task in libero_task_map[libero_suite]:
        language = grab_language_from_filename(task + ".bddl")
        task_maps[libero_suite][task] = Task(
            name=task,
            language=language,
            problem="Libero",
            problem_folder=libero_suite,
            bddl_file=f"{task}.bddl",
            init_states_file=f"{task}.pruned_init",
        )

        # print(language, "\n", f"{task}.bddl", "\n")
        # print("")

# yy: add my task
yy_suites = [
    # "yy_try",
    "modified_libero",
    "single_step"
]
# yy: if you wanna the task description the same as the original one, set True here.
keep_language_unchanged = True
for yy_suite in yy_suites:
    task_maps[yy_suite] = {}

    for task in yy_task_map[yy_suite]:
        if keep_language_unchanged:
            mapping_pth = f"libero/mappings/{yy_suite}.json"
            with open(mapping_pth, 'r') as json_file:
                mapping = json.load(json_file)
            task_ori = find_keys_by_value(mapping, task)
            language = grab_language_from_filename(task_ori + ".bddl", is_yy=True)
        else:
            language = grab_language_from_filename(task + ".bddl", is_yy=True)
        task_maps[yy_suite][task] = Task(
            name=task,
            language=language,
            problem="Libero",
            problem_folder=yy_suite,
            bddl_file=f"{task}.bddl",
            init_states_file=f"{task}.pruned_init",
        )




task_orders = [
    # train skills (0 ~ 3)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
    # [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],  # transformer
    [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],  # transformer
    # [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],  # rnn
    [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],  # rnn
    # [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],  # vilt
    [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],  # vilt
    # eval_modified_env (4)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    # eval ori env (5)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    # my method - 1: eval modified env (with midify_back) (6)
    [1, 11, 18], # TODO
    # eval long horizon tasks (7 ~ 16)
    [2, 3, 5],
    [3, 2, 5],
    [6, 7, 10],
    [6, 8, 10],
    [19, 18, 16],
    [19, 16, 18],
    [17, 16, 20],
    [20, 17, 16],
    [32, 35, 36],
    [34, 35, 36],

    # old ones
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [4, 6, 8, 7, 3, 1, 2, 0, 9, 5],
    [6, 3, 5, 0, 4, 2, 9, 1, 8, 7],
    [7, 4, 3, 0, 8, 1, 2, 5, 9, 6],
    [4, 5, 6, 3, 8, 0, 2, 7, 1, 9],
    [1, 2, 3, 0, 6, 9, 5, 7, 4, 8],
    [3, 7, 8, 1, 6, 2, 9, 4, 0, 5],
    [4, 2, 9, 7, 6, 8, 5, 1, 3, 0],
    [1, 8, 5, 4, 0, 9, 6, 7, 2, 3],
    [8, 3, 6, 4, 9, 5, 1, 2, 0, 7],
    [6, 9, 0, 5, 7, 1, 2, 8, 3, 4],
    [6, 8, 3, 1, 0, 2, 5, 9, 7, 4],
    [8, 0, 6, 9, 4, 1, 7, 3, 2, 5],
    [3, 8, 6, 4, 2, 5, 0, 7, 1, 9],
    [7, 1, 5, 6, 3, 2, 8, 9, 4, 0],
    [2, 0, 9, 5, 3, 6, 8, 7, 1, 4],
    [3, 5, 9, 6, 2, 4, 8, 7, 1, 0],
    [7, 6, 5, 9, 0, 3, 4, 2, 8, 1],
    [2, 5, 0, 9, 3, 1, 6, 4, 8, 7],
    [3, 5, 1, 2, 7, 8, 6, 0, 4, 9],
    [3, 4, 1, 9, 7, 6, 8, 2, 0, 5],
]


class Benchmark(abc.ABC):
    """A Benchmark."""

    def __init__(self, task_order_index=0, n_tasks_=None):
        self.task_embs = None
        self.task_order_index = task_order_index
        self.n_tasks_ = n_tasks_
        print(f"[INFO] Benchmark task order index: {self.task_order_index}")

    def _make_benchmark(self):
        tasks = list(task_maps[self.name].values())
        # yy: I changed this
        # if (self.name == "libero_90") or (self.name == "yy_try") or (self.name == "modified_libero"):
        #     self.tasks = tasks
        if (self.name == "yy_try"):
            self.tasks = tasks
        else:
            print(f"[info] using task orders {task_orders[self.task_order_index]}")
            self.tasks = [tasks[i] for i in task_orders[self.task_order_index]]
        # yy: set 1 for just traininig 1 task
        if self.n_tasks_:
            self.n_tasks = self.n_tasks_
        else:
            # if n_tasks_ set to None, it means to use all tasks
            self.n_tasks = len(self.tasks)

    def get_num_tasks(self):
        return self.n_tasks

    def get_task_names(self):
        return [task.name for task in self.tasks]

    def get_task_problems(self):
        return [task.problem for task in self.tasks]

    def get_task_bddl_files(self):
        return [task.bddl_file for task in self.tasks]

    def get_task_bddl_file_path(self, i):
        bddl_file_path = os.path.join(
            get_libero_path("bddl_files"),
            self.tasks[i].problem_folder,
            self.tasks[i].bddl_file,
        )
        return bddl_file_path

    def get_task_demonstration(self, i):
        assert (
            0 <= i and i < self.n_tasks
        ), f"[error] task number {i} is outer of range {self.n_tasks}"
        # this path is relative to the datasets folder
        demo_path = f"{self.tasks[i].problem_folder}/{self.tasks[i].name}_demo.hdf5"
        return demo_path

    def get_task(self, i):
        return self.tasks[i]

    def get_task_emb(self, i):
        return self.task_embs[i]

    def get_task_init_states(self, i):
        init_states_path = os.path.join(
            get_libero_path("init_states"),
            self.tasks[i].problem_folder,
            self.tasks[i].init_states_file,
        )
        init_states = torch.load(init_states_path)
        return init_states

    def set_task_embs(self, task_embs):
        self.task_embs = task_embs

@register_benchmark
class LIBERO_SPATIAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial"
        self._make_benchmark()


@register_benchmark
class LIBERO_OBJECT(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object"
        self._make_benchmark()


@register_benchmark
class LIBERO_GOAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal"
        self._make_benchmark()


@register_benchmark
class LIBERO_90(Benchmark):
    def __init__(self, task_order_index=0, n_tasks_=None):
        super().__init__(task_order_index=task_order_index, n_tasks_=n_tasks_)
        # yy: I comment this
        # assert (
        #     task_order_index == 0
        # ), "[error] currently only support task order for 10-task suites"
        self.name = "libero_90"
        self._make_benchmark()


@register_benchmark
class LIBERO_10(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10"
        self._make_benchmark()


@register_benchmark
class LIBERO_100(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_100"
        self._make_benchmark()


@register_benchmark
class yy_try(Benchmark):
    def __init__(self, task_order_index=0, n_tasks_=1):
        super().__init__(task_order_index=task_order_index, n_tasks_=n_tasks_)
        assert (
            task_order_index == 0
        ), "[error] currently only support task order for 10-task suites"
        self.name = "yy_try"
        self._make_benchmark()


@register_benchmark
class modified_libero(Benchmark):
    def __init__(self, task_order_index=0, n_tasks_=1):
        super().__init__(task_order_index=task_order_index, n_tasks_=n_tasks_)
        self.name = "modified_libero"
        self._make_benchmark()


@register_benchmark
class single_step(Benchmark):
    def __init__(self, task_order_index=0, n_tasks_=1):
        super().__init__(task_order_index=task_order_index, n_tasks_=n_tasks_)
        self.name = "single_step"
        self._make_benchmark()