# TensorCFR

[![pipeline status](https://gitlab.com/beyond-deepstack/TensorCFR/badges/master/pipeline.svg)](https://gitlab.com/beyond-deepstack/TensorCFR/pipelines)
[![coverage report](https://gitlab.com/beyond-deepstack/TensorCFR/badges/master/coverage.svg)](https://gitlab.com/beyond-deepstack/TensorCFR/commits/master)

an implementation of [CFR+](https://arxiv.org/abs/1407.5042) with TensorFlow tensors (for GPU)

## Profiling

(by @janrudolf from [#66](https://gitlab.com/beyond-deepstack/TensorCFR/issues/66))

![Screenshot of profiling in TensorBoard](./doc/TensorBoard_profiling.png)

1) To see computation time and memory consumption in TensorBoard:

   1.  Run the `tensorcfr.py`. 
   2.  Run TensorBoard. 
   3.  As in the picture, choose the latest run of `tensorcfr.py` in the rolldown menu named `Run`.
   4.  Choose the name with `*,with_time_mem`. 
   5.  After that, you can choose which step you want, then you are able to choose `Compute time` and click on the node to investigate it.

2) To compute the total compute time per one CFR step/iteration, the [TF Profiler](https://www.tensorflow.org/api_docs/python/tf/profiler/Profiler) is used.

It prints (for `profiling=True`) a table as command line output like this (accelerator = gpu/tpu):

```
==================Model Analysis Report======================
node name | requested bytes | total execution time | accelerator execution time | cpu execution time
_TFProfRoot (--/4.38KB, --/1.04ms, --/0us, --/1.04ms)
  domain_definitions (0B/1.48KB, 0us/280us, 0us/0us, 0us/280us)
    domain_definitions/NotEqual_1 (4B/4B, 21us/21us, 0us/0us, 21us/21us)
    domain_definitions/signum_of_current_player (4B/12B, 12us/20us, 0us/0us, 12us/20us)
      domain_definitions/signum_of_current_player/e (4B/4B, 4us/4us, 0us/0us, 4us/4us)
      domain_definitions/signum_of_current_player/t (4B/4B, 4us/4us, 0us/0us, 4us/4us)
    domain_definitions/Select_1 (120B/120B, 13us/13us, 0us/0us, 13us/13us)
    domain_definitions/cumulative_infoset_strategies_lvl0 (20B/20B, 13us/13us, 0us/0us, 13us/13us)
    domain_definitions/NotEqual (1B/5B, 7us/11us, 0us/0us, 7us/11us)
      domain_definitions/NotEqual/y (4B/4B, 4us/4us, 0us/0us, 4us/4us)
    domain_definitions/Select (60B/60B, 11us/11us, 0us/0us, 11us/11us)
    domain_definitions/node_to_infoset_lvl0 (4B/4B, 9us/9us, 0us/0us, 9us/9us)
    domain_definitions/current_opponent (4B/4B, 8us/8us, 0us/0us, 8us/8us)
    domain_definitions/Equal_2 (1B/1B, 7us/7us, 0us/0us, 7us/7us)
    domain_definitions/current_updating_player (4B/4B, 7us/7us, 0us/0us, 7us/7us)
    domain_definitions/cumulative_infoset_strategies_lvl2 (72B/72B, 7us/7us, 0us/0us, 7us/7us)
    domain_definitions/cumulative_infoset_strategies_lvl1 (48B/48B, 7us/7us, 0us/0us, 7us/7us)
    domain_definitions/node_to_infoset_lvl2 (60B/60B, 7us/7us, 0us/0us, 7us/7us)
    domain_definitions/Variable_2 (120B/120B, 7us/7us, 0us/0us, 7us/7us)
    domain_definitions/positive_cumulative_regrets_lvl0 (20B/20B, 7us/7us, 0us/0us, 7us/7us)
    domain_definitions/positive_cumulative_regrets_lvl1 (48B/48B, 7us/7us, 0us/0us, 7us/7us)
    domain_definitions/NotEqual_2 (9B/9B, 7us/7us, 0us/0us, 7us/7us)
```