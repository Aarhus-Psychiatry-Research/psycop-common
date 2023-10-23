# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/Aarhus-Psychiatry-Research/psycop-common/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                                                            |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| psycop/\_\_init\_\_.py                                                                                          |        0 |        0 |    100% |           |
| psycop/common/cohort\_definition.py                                                                             |       45 |       15 |     67% |18, 31, 35, 47, 52, 60-80 |
| psycop/common/data\_structures/\_\_init\_\_.py                                                                  |        3 |        0 |    100% |           |
| psycop/common/data\_structures/patient.py                                                                       |       57 |        5 |     91% |11-14, 35, 68 |
| psycop/common/data\_structures/prediction\_time.py                                                              |       11 |        2 |     82% |       7-9 |
| psycop/common/data\_structures/static\_feature.py                                                               |        9 |        1 |     89% |         7 |
| psycop/common/data\_structures/temporal\_event.py                                                               |       11 |        1 |     91% |         7 |
| psycop/common/feature\_generation/\_\_init\_\_.py                                                               |        0 |        0 |    100% |           |
| psycop/common/feature\_generation/application\_modules/\_\_init\_\_.py                                          |        0 |        0 |    100% |           |
| psycop/common/feature\_generation/application\_modules/chunked\_feature\_generation.py                          |       50 |       19 |     62% |31-57, 64-69, 98-100, 109, 118-121 |
| psycop/common/feature\_generation/application\_modules/filter\_prediction\_times.py                             |       44 |        3 |     93% | 9, 45, 81 |
| psycop/common/feature\_generation/application\_modules/flatten\_dataset.py                                      |       27 |        6 |     78% |22-28, 113 |
| psycop/common/feature\_generation/application\_modules/project\_setup.py                                        |       29 |        5 |     83% |     53-70 |
| psycop/common/feature\_generation/application\_modules/save\_dataset\_to\_disk.py                               |       50 |       14 |     72% |29-35, 79-83, 107-112, 117 |
| psycop/common/feature\_generation/application\_modules/wandb\_utils.py                                          |       10 |        3 |     70% |     13-15 |
| psycop/common/feature\_generation/data\_checks/\_\_init\_\_.py                                                  |        0 |        0 |    100% |           |
| psycop/common/feature\_generation/data\_checks/flattened/feature\_describer.py                                  |       70 |       18 |     74% |20-21, 42, 55-57, 75-90, 100, 167-168, 230 |
| psycop/common/feature\_generation/data\_checks/raw/check\_raw\_df.py                                            |       63 |       10 |     84% |8-10, 86, 170-175, 185, 216 |
| psycop/common/feature\_generation/data\_checks/utils.py                                                         |       16 |        3 |     81% |   7-9, 72 |
| psycop/common/feature\_generation/loaders/\_\_init\_\_.py                                                       |        0 |        0 |    100% |           |
| psycop/common/feature\_generation/loaders/filters/\_\_init\_\_.py                                               |        0 |        0 |    100% |           |
| psycop/common/feature\_generation/loaders/filters/cvd\_filters.py                                               |       12 |        0 |    100% |           |
| psycop/common/feature\_generation/loaders/filters/diabetes\_filters.py                                          |       16 |        4 |     75% |  6, 49-56 |
| psycop/common/feature\_generation/loaders/flattened/\_\_init\_\_.py                                             |        1 |        0 |    100% |           |
| psycop/common/feature\_generation/loaders/flattened/local\_feature\_loaders.py                                  |       18 |        7 |     61% |9-11, 27-30, 76, 98, 118 |
| psycop/common/feature\_generation/loaders/non\_numerical\_coercer.py                                            |       15 |        1 |     93% |        11 |
| psycop/common/feature\_generation/loaders/raw/\_\_init\_\_.py                                                   |       10 |        0 |    100% |           |
| psycop/common/feature\_generation/loaders/raw/load\_coercion.py                                                 |      102 |       52 |     49% |33-75, 97-118, 127-136, 151, 166, 180, 194, 208, 224-233, 248-263, 277-289, 304-324, 342, 356, 370, 384, 398, 412, 426, 440, 454, 468, 482 |
| psycop/common/feature\_generation/loaders/raw/load\_demographic.py                                              |       21 |       13 |     38% |13-25, 30-43 |
| psycop/common/feature\_generation/loaders/raw/load\_diagnoses.py                                                |      211 |       83 |     61% |22, 58-101, 112, 131, 153, 172, 191, 210, 229, 248-277, 287-301, 311-328, 338-351, 361-381, 392-421, 434, 453, 472, 491, 511, 530, 549, 568, 587, 606, 625, 644, 663, 685, 704, 723, 742, 764, 783, 802, 821, 843, 862, 881, 900, 922, 941, 960, 979, 999, 1018, 1037, 1056, 1075, 1094, 1116, 1135, 1154, 1173, 1192, 1212, 1231, 1250, 1270, 1289, 1308, 1327, 1347 |
| psycop/common/feature\_generation/loaders/raw/load\_ids.py                                                      |       15 |        5 |     67% | 11, 30-36 |
| psycop/common/feature\_generation/loaders/raw/load\_lab\_results.py                                             |      180 |       92 |     49% |31-54, 73-95, 113-139, 157-180, 198-238, 249, 261-308, 320, 332, 344-351, 363, 375, 387, 399, 411, 423, 435, 447, 459, 471, 483, 495, 507, 519, 531, 543, 555, 567, 579, 591, 603, 615, 627, 639, 651, 663, 675, 687, 699, 711, 723, 735, 747, 762 |
| psycop/common/feature\_generation/loaders/raw/load\_medications.py                                              |      201 |       77 |     62% |53-113, 135-144, 168, 189, 217, 254, 279, 303, 327, 351, 374, 397, 426, 452, 476, 502, 521, 540, 559, 578, 597, 617, 638, 658, 679, 698, 717, 750, 774, 797, 817, 837, 857, 876, 895, 914, 933, 952, 971, 990, 1010, 1030, 1049, 1068, 1087, 1106, 1125, 1144, 1163, 1182, 1201, 1220, 1239, 1258, 1277, 1296, 1315, 1334, 1353, 1373 |
| psycop/common/feature\_generation/loaders/raw/load\_moves.py                                                    |       16 |       10 |     38% |12-18, 24-30, 34 |
| psycop/common/feature\_generation/loaders/raw/load\_structured\_sfi.py                                          |       74 |       44 |     41% |11, 32-59, 64, 76-92, 97-114, 119, 129, 139, 149-158, 163-172, 177-187, 192-206, 211-224, 229-239 |
| psycop/common/feature\_generation/loaders/raw/load\_t2d\_outcomes.py                                            |       19 |       11 |     42% |13-24, 29-42 |
| psycop/common/feature\_generation/loaders/raw/load\_text.py                                                     |       54 |       33 |     39% |18-20, 32, 77-87, 113-146, 167-183, 200, 219, 240, 265-277 |
| psycop/common/feature\_generation/loaders/raw/load\_visits.py                                                   |       77 |       46 |     40% |65-169, 178, 192-203, 212, 228, 247-258, 269, 286, 296-308 |
| psycop/common/feature\_generation/loaders/raw/sql\_load.py                                                      |       21 |       13 |     38% |     42-70 |
| psycop/common/feature\_generation/loaders/raw/utils.py                                                          |       74 |       55 |     26% |27-38, 55-77, 133-291 |
| psycop/common/feature\_generation/sequences/cohort\_definer\_to\_prediction\_times.py                           |       40 |        3 |     92% |   102-115 |
| psycop/common/feature\_generation/sequences/event\_dataframes\_to\_patient.py                                   |       67 |        0 |    100% |           |
| psycop/common/feature\_generation/sequences/patient\_loaders.py                                                 |       45 |       14 |     69% |21, 33-35, 77, 96-98, 106-122, 126 |
| psycop/common/feature\_generation/sequences/utils\_for\_testing.py                                              |        5 |        0 |    100% |           |
| psycop/common/feature\_generation/text\_models/fit\_text\_models.py                                             |       10 |        1 |     90% |        34 |
| psycop/common/feature\_generation/text\_models/preprocessing.py                                                 |       22 |        5 |     77% |     71-91 |
| psycop/common/feature\_generation/text\_models/utils.py                                                         |       12 |        4 |     67% |23-24, 37-38 |
| psycop/common/feature\_generation/utils.py                                                                      |       48 |       23 |     52% |16, 37, 64, 68, 72, 93-100, 113, 133, 137, 147-162 |
| psycop/common/global\_utils/cache.py                                                                            |        8 |        1 |     88% |         8 |
| psycop/common/global\_utils/paths.py                                                                            |        6 |        0 |    100% |           |
| psycop/common/global\_utils/pickle.py                                                                           |       11 |        6 |     45% |7-10, 14-17 |
| psycop/common/global\_utils/pydantic\_basemodel.py                                                              |       18 |        1 |     94% |        25 |
| psycop/common/global\_utils/synth\_data\_generator/\_\_init\_\_.py                                              |        0 |        0 |    100% |           |
| psycop/common/global\_utils/synth\_data\_generator/synth\_col\_generators.py                                    |       64 |       21 |     67% |70-98, 127-136, 139, 149, 174, 242-256 |
| psycop/common/global\_utils/synth\_data\_generator/synth\_prediction\_times\_generator.py                       |       18 |        0 |    100% |           |
| psycop/common/global\_utils/synth\_data\_generator/utils.py                                                     |       11 |        0 |    100% |           |
| psycop/common/minimal\_pipeline.py                                                                              |       12 |        0 |    100% |           |
| psycop/common/model\_evaluation/\_\_init\_\_.py                                                                 |        0 |        0 |    100% |           |
| psycop/common/model\_evaluation/binary/bootstrap\_estimates.py                                                  |       18 |        3 |     83% |     35-37 |
| psycop/common/model\_evaluation/binary/global\_performance/roc\_auc.py                                          |       35 |        1 |     97% |        60 |
| psycop/common/model\_evaluation/binary/performance\_by\_ppr/performance\_by\_ppr.py                             |       55 |        0 |    100% |           |
| psycop/common/model\_evaluation/binary/performance\_by\_ppr/prop\_of\_all\_events\_hit\_by\_true\_positive.py   |        9 |        0 |    100% |           |
| psycop/common/model\_evaluation/binary/subgroup\_data.py                                                        |       15 |        1 |     93% |        47 |
| psycop/common/model\_evaluation/binary/time/absolute\_data.py                                                   |        8 |        0 |    100% |           |
| psycop/common/model\_evaluation/binary/time/timedelta\_data.py                                                  |       37 |        6 |     84% |39, 106, 149-171 |
| psycop/common/model\_evaluation/binary/utils.py                                                                 |       35 |        6 |     83% |21, 66, 110-116 |
| psycop/common/model\_evaluation/confusion\_matrix/confusion\_matrix.py                                          |       27 |        0 |    100% |           |
| psycop/common/model\_evaluation/markdown/md\_objects.py                                                         |       56 |        5 |     91% |28, 83-88, 138 |
| psycop/common/model\_evaluation/patchwork/patchwork\_grid.py                                                    |       35 |        0 |    100% |           |
| psycop/common/model\_evaluation/utils.py                                                                        |      100 |       44 |     56% |55, 84-95, 115, 131-134, 162, 179, 233-239, 251-252, 266-273, 285, 303-306, 313-318, 326-336, 341, 349-356 |
| psycop/common/model\_training/\_\_init\_\_.py                                                                   |        0 |        0 |    100% |           |
| psycop/common/model\_training/application\_modules/\_\_init\_\_.py                                              |        0 |        0 |    100% |           |
| psycop/common/model\_training/application\_modules/train\_model/\_\_init\_\_.py                                 |        0 |        0 |    100% |           |
| psycop/common/model\_training/application\_modules/train\_model/main.py                                         |       38 |        0 |    100% |           |
| psycop/common/model\_training/application\_modules/wandb\_handler.py                                            |       34 |        2 |     94% |    47, 69 |
| psycop/common/model\_training/config\_schemas/\_\_init\_\_.py                                                   |        0 |        0 |    100% |           |
| psycop/common/model\_training/config\_schemas/conf\_utils.py                                                    |       42 |        5 |     88% |66-72, 95, 108, 113 |
| psycop/common/model\_training/config\_schemas/data.py                                                           |       22 |        0 |    100% |           |
| psycop/common/model\_training/config\_schemas/debug.py                                                          |        5 |        0 |    100% |           |
| psycop/common/model\_training/config\_schemas/full\_config.py                                                   |       16 |        0 |    100% |           |
| psycop/common/model\_training/config\_schemas/model.py                                                          |        5 |        0 |    100% |           |
| psycop/common/model\_training/config\_schemas/preprocessing.py                                                  |       29 |        0 |    100% |           |
| psycop/common/model\_training/config\_schemas/project.py                                                        |       12 |        0 |    100% |           |
| psycop/common/model\_training/config\_schemas/train.py                                                          |        5 |        0 |    100% |           |
| psycop/common/model\_training/data\_loader/\_\_init\_\_.py                                                      |        0 |        0 |    100% |           |
| psycop/common/model\_training/data\_loader/col\_name\_checker.py                                                |       29 |        0 |    100% |           |
| psycop/common/model\_training/data\_loader/data\_loader.py                                                      |       82 |        8 |     90% |43, 149, 155, 163, 188, 202-205, 225 |
| psycop/common/model\_training/data\_loader/tests/conftest.py                                                    |       15 |        0 |    100% |           |
| psycop/common/model\_training/data\_loader/utils.py                                                             |       33 |       15 |     55% |23, 81-102 |
| psycop/common/model\_training/preprocessing/\_\_init\_\_.py                                                     |        0 |        0 |    100% |           |
| psycop/common/model\_training/preprocessing/post\_split/\_\_init\_\_.py                                         |        0 |        0 |    100% |           |
| psycop/common/model\_training/preprocessing/post\_split/create\_pipeline.py                                     |       31 |        5 |     84% |22, 60, 77-81, 115 |
| psycop/common/model\_training/preprocessing/post\_split/pipeline.py                                             |       15 |        0 |    100% |           |
| psycop/common/model\_training/preprocessing/pre\_split/\_\_init\_\_.py                                          |        0 |        0 |    100% |           |
| psycop/common/model\_training/preprocessing/pre\_split/full\_processor.py                                       |       34 |        0 |    100% |           |
| psycop/common/model\_training/preprocessing/pre\_split/processors/col\_filter.py                                |       85 |       17 |     80% |47, 65-76, 145, 174-190, 209, 225 |
| psycop/common/model\_training/preprocessing/pre\_split/processors/row\_filter.py                                |       81 |       11 |     86% |49, 85, 118, 132-137, 157, 167, 174, 183-187 |
| psycop/common/model\_training/preprocessing/pre\_split/processors/value\_cleaner.py                             |       45 |        1 |     98% |       107 |
| psycop/common/model\_training/preprocessing/pre\_split/processors/value\_transformer.py                         |       39 |       13 |     67% |40, 46-56, 65-76, 84, 87 |
| psycop/common/model\_training/tests/\_\_init\_\_.py                                                             |        0 |        0 |    100% |           |
| psycop/common/model\_training/tests/test\_data/\_\_init\_\_.py                                                  |        0 |        0 |    100% |           |
| psycop/common/model\_training/training/\_\_init\_\_.py                                                          |        0 |        0 |    100% |           |
| psycop/common/model\_training/training/model\_specs.py                                                          |       13 |        0 |    100% |           |
| psycop/common/model\_training/training/train\_and\_predict.py                                                   |      111 |       13 |     88% |105, 302-319, 374 |
| psycop/common/model\_training/training/utils.py                                                                 |       14 |        2 |     86% |    23, 33 |
| psycop/common/model\_training/training\_output/\_\_init\_\_.py                                                  |        0 |        0 |    100% |           |
| psycop/common/model\_training/training\_output/artifact\_saver/to\_disk.py                                      |       53 |        1 |     98% |        58 |
| psycop/common/model\_training/training\_output/dataclasses.py                                                   |       57 |       12 |     79% |82-85, 94-103, 106 |
| psycop/common/model\_training/training\_output/model\_evaluator.py                                              |       33 |        1 |     97% |        41 |
| psycop/common/model\_training/utils/\_\_init\_\_.py                                                             |        0 |        0 |    100% |           |
| psycop/common/model\_training/utils/col\_name\_inference.py                                                     |       41 |        9 |     78% |35, 67-68, 76, 96-101, 128 |
| psycop/common/model\_training/utils/decorators.py                                                               |       43 |        1 |     98% |        31 |
| psycop/common/model\_training/utils/utils.py                                                                    |       89 |       29 |     67% |40, 119-122, 148, 153-154, 162, 167, 216-222, 234-235, 252, 256, 286-289, 297-301, 309-319, 331-335 |
| psycop/common/sequence\_models/\_\_init\_\_.py                                                                  |        6 |        0 |    100% |           |
| psycop/common/sequence\_models/aggregators.py                                                                   |       12 |        1 |     92% |        14 |
| psycop/common/sequence\_models/dataset.py                                                                       |       20 |        0 |    100% |           |
| psycop/common/sequence\_models/embedders/BEHRT\_embedders.py                                                    |      151 |        3 |     98% |75, 141, 177 |
| psycop/common/sequence\_models/embedders/interface.py                                                           |       20 |        5 |     75% |31, 34, 37, 43, 49 |
| psycop/common/sequence\_models/tasks.py                                                                         |      142 |       12 |     92% |226-229, 232-234, 244-245, 320-330 |
| psycop/common/sequence\_models/tests/\_\_init\_\_.py                                                            |        0 |        0 |    100% |           |
| psycop/common/sequence\_models/tests/conftest.py                                                                |       31 |        7 |     77% |     66-85 |
| psycop/common/test\_utils/str\_to\_df.py                                                                        |       31 |        1 |     97% |        80 |
| psycop/common/test\_utils/test\_data/model\_eval/generate\_synthetic\_dataset\_for\_eval.py                     |       55 |       39 |     29% |37, 42, 61-65, 84-88, 92-170 |
| psycop/conftest.py                                                                                              |       47 |        3 |     94% |33, 37, 104 |
| psycop/projects/forced\_admission\_inpatient/utils/feature\_name\_to\_readable.py                               |       26 |        5 |     81% | 15, 39-42 |
| psycop/projects/restraint/\_\_init\_\_.py                                                                       |        0 |        0 |    100% |           |
| psycop/projects/restraint/model\_evaluation/config.py                                                           |       40 |        0 |    100% |           |
| psycop/projects/restraint/model\_evaluation/figures/feature\_importance/shap/get\_shap\_values.py               |       73 |       41 |     44% |26-37, 42-54, 70, 78-92, 102-134, 164-172, 181-194 |
| psycop/projects/restraint/model\_evaluation/figures/feature\_importance/shap/shap\_plots.py                     |       53 |       29 |     45% |41, 49, 75-87, 96-102, 118-148 |
| psycop/projects/restraint/model\_evaluation/figures/feature\_importance/shap/shap\_table.py                     |        7 |        0 |    100% |           |
| psycop/projects/restraint/model\_evaluation/utils/feature\_name\_to\_readable.py                                |       53 |       11 |     79% |6-19, 74-82, 145 |
| psycop/projects/restraint/test/\_\_init\_\_.py                                                                  |        0 |        0 |    100% |           |
| psycop/projects/restraint/test/test\_model\_evaluation/\_\_init\_\_.py                                          |        0 |        0 |    100% |           |
| psycop/projects/restraint/test/test\_model\_evaluation/conftest.py                                              |        7 |        0 |    100% |           |
| psycop/projects/restraint/utils/best\_runs.py                                                                   |       75 |       32 |     57% |24, 30-38, 42-48, 51-53, 70-74, 77, 80, 86, 90, 94, 98-99, 104-106, 109-113, 117, 121-122, 130 |
| psycop/projects/restraint/utils/feature\_name\_to\_readable.py                                                  |       26 |       22 |     15% |6-16, 21-54, 59-64 |
| psycop/projects/sequence\_models/train.py                                                                       |       89 |       19 |     79% |113-116, 191-241 |
| psycop/projects/t2d/feature\_generation/\_\_init\_\_.py                                                         |        0 |        0 |    100% |           |
| psycop/projects/t2d/feature\_generation/cohort\_definition/eligible\_prediction\_times/add\_age.py              |        9 |        5 |     44% |     10-18 |
| psycop/projects/t2d/feature\_generation/cohort\_definition/eligible\_prediction\_times/eligible\_config.py      |        4 |        0 |    100% |           |
| psycop/projects/t2d/feature\_generation/cohort\_definition/eligible\_prediction\_times/single\_filters.py       |       46 |       23 |     50% |29-30, 36-38, 44-72, 78-99, 105-115 |
| psycop/projects/t2d/feature\_generation/cohort\_definition/outcome\_specification/combined.py                   |       15 |        8 |     47% | 18-42, 46 |
| psycop/projects/t2d/feature\_generation/cohort\_definition/outcome\_specification/lab\_results.py               |       27 |       17 |     37% |16-19, 23, 27, 35, 39, 47-52, 64-74, 78 |
| psycop/projects/t2d/feature\_generation/cohort\_definition/outcome\_specification/medications.py                |       12 |        7 |     42% |9-16, 20-30, 34-36 |
| psycop/projects/t2d/feature\_generation/cohort\_definition/outcome\_specification/t1d\_diagnoses.py             |        9 |        5 |     44% |7-16, 20-22 |
| psycop/projects/t2d/feature\_generation/cohort\_definition/outcome\_specification/t2d\_diagnoses.py             |        9 |        5 |     44% |7-16, 20-22 |
| psycop/projects/t2d/feature\_generation/cohort\_definition/t2d\_cohort\_definer.py                              |       16 |        5 |     69% |26-33, 47, 51-53 |
| psycop/projects/t2d/paper\_outputs/config.py                                                                    |       18 |        0 |    100% |           |
| psycop/projects/t2d/paper\_outputs/intervention\_eval/hba1c.py                                                  |       20 |        9 |     55% |     36-75 |
| psycop/projects/t2d/paper\_outputs/model\_description/feature\_importance/shap/conftest.py                      |        7 |        0 |    100% |           |
| psycop/projects/t2d/paper\_outputs/model\_description/feature\_importance/shap/get\_shap\_values.py             |       59 |       34 |     42% |18-29, 34-46, 62, 70-84, 93-125, 153-163 |
| psycop/projects/t2d/paper\_outputs/model\_description/feature\_importance/shap/plot\_shap.py                    |       31 |        0 |    100% |           |
| psycop/projects/t2d/paper\_outputs/model\_description/feature\_importance/shap/shap\_table.py                   |        7 |        0 |    100% |           |
| psycop/projects/t2d/paper\_outputs/model\_description/performance/auroc.py                                      |       12 |        7 |     42% |10-19, 23-25 |
| psycop/projects/t2d/paper\_outputs/model\_description/performance/confusion\_matrix\_pipeline.py                |       15 |        8 |     47% |14-33, 37-39 |
| psycop/projects/t2d/paper\_outputs/model\_description/performance/incidence\_by\_time\_until\_diagnosis.py      |       17 |       10 |     41% |14-54, 58-60 |
| psycop/projects/t2d/paper\_outputs/model\_description/performance/main\_performance\_figure.py                  |       11 |        3 |     73% | 20, 33-35 |
| psycop/projects/t2d/paper\_outputs/model\_description/performance/performance\_by\_ppr.py                       |       31 |        7 |     77% |72-84, 88-90 |
| psycop/projects/t2d/paper\_outputs/model\_description/performance/plotnine\_confusion\_matrix.py                |       12 |        0 |    100% |           |
| psycop/projects/t2d/paper\_outputs/model\_description/performance/sensitivity\_by\_time\_to\_event\_pipeline.py |       35 |       20 |     43% |48-56, 60-90, 94-100, 104-106 |
| psycop/projects/t2d/paper\_outputs/model\_description/robustness/robustness\_plot.py                            |       12 |        1 |     92% |        51 |
| psycop/projects/t2d/paper\_outputs/model\_permutation/boolean\_features.py                                      |       27 |       13 |     52% |32, 42-60, 64-73 |
| psycop/projects/t2d/paper\_outputs/model\_permutation/modified\_dataset.py                                      |       44 |       29 |     34% |19, 30, 37-44, 53-100 |
| psycop/projects/t2d/paper\_outputs/model\_permutation/only\_hba1c.py                                            |       43 |       20 |     53% |38-56, 91-117 |
| psycop/projects/t2d/paper\_outputs/utils/create\_patchwork\_figure.py                                           |       30 |       22 |     27% |     22-58 |
| psycop/projects/t2d/utils/feature\_name\_to\_readable.py                                                        |       26 |        5 |     81% | 15, 44-47 |
| psycop/projects/t2d/utils/pipeline\_objects.py                                                                  |      121 |       62 |     49% |19-20, 28, 50, 54-62, 66-72, 75-77, 96-110, 113-117, 120, 123, 137-139, 152-156, 159-161, 165, 177-184, 197-204, 221-236, 247 |
|                                                                                                       **TOTAL** | **5299** | **1496** | **72%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/Aarhus-Psychiatry-Research/psycop-common/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/Aarhus-Psychiatry-Research/psycop-common/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Aarhus-Psychiatry-Research/psycop-common/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/Aarhus-Psychiatry-Research/psycop-common/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FAarhus-Psychiatry-Research%2Fpsycop-common%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/Aarhus-Psychiatry-Research/psycop-common/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.