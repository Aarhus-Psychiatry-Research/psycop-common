# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/Aarhus-Psychiatry-Research/psycop-common/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                                                            |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| psycop/\_\_init\_\_.py                                                                                          |        0 |        0 |    100% |           |
| psycop/common/cohort\_definition.py                                                                             |       43 |       13 |     70% |18, 31, 35, 47, 52, 60-77 |
| psycop/common/data\_structures/patient.py                                                                       |       35 |        2 |     94% |    22, 55 |
| psycop/common/data\_structures/prediction\_time.py                                                              |       16 |        5 |     69% |      7-14 |
| psycop/common/data\_structures/static\_feature.py                                                               |        9 |        1 |     89% |         7 |
| psycop/common/data\_structures/temporal\_event.py                                                               |       11 |        1 |     91% |         7 |
| psycop/common/feature\_generation/\_\_init\_\_.py                                                               |        0 |        0 |    100% |           |
| psycop/common/feature\_generation/application\_modules/\_\_init\_\_.py                                          |        0 |        0 |    100% |           |
| psycop/common/feature\_generation/application\_modules/chunked\_feature\_generation.py                          |       50 |       19 |     62% |31-57, 64-69, 98-100, 109, 118-121 |
| psycop/common/feature\_generation/application\_modules/filter\_prediction\_times.py                             |       44 |        3 |     93% | 9, 45, 81 |
| psycop/common/feature\_generation/application\_modules/flatten\_dataset.py                                      |       27 |        6 |     78% |22-28, 113 |
| psycop/common/feature\_generation/application\_modules/project\_setup.py                                        |       29 |        5 |     83% |     53-70 |
| psycop/common/feature\_generation/application\_modules/save\_dataset\_to\_disk.py                               |       41 |        8 |     80% |29-35, 79-83, 105 |
| psycop/common/feature\_generation/application\_modules/wandb\_utils.py                                          |       10 |        3 |     70% |     13-15 |
| psycop/common/feature\_generation/data\_checks/\_\_init\_\_.py                                                  |        0 |        0 |    100% |           |
| psycop/common/feature\_generation/data\_checks/flattened/feature\_describer.py                                  |       70 |       18 |     74% |20-21, 42, 55-57, 75-90, 100, 167-168, 230 |
| psycop/common/feature\_generation/data\_checks/raw/check\_raw\_df.py                                            |       63 |       10 |     84% |8-10, 86, 170-175, 185, 216 |
| psycop/common/feature\_generation/data\_checks/utils.py                                                         |       16 |        3 |     81% |   7-9, 72 |
| psycop/common/feature\_generation/loaders/\_\_init\_\_.py                                                       |        0 |        0 |    100% |           |
| psycop/common/feature\_generation/loaders/filters/\_\_init\_\_.py                                               |        0 |        0 |    100% |           |
| psycop/common/feature\_generation/loaders/filters/diabetes\_filters.py                                          |       14 |        4 |     71% |  6, 41-48 |
| psycop/common/feature\_generation/loaders/flattened/\_\_init\_\_.py                                             |        1 |        0 |    100% |           |
| psycop/common/feature\_generation/loaders/flattened/local\_feature\_loaders.py                                  |       18 |        7 |     61% |9-11, 27-30, 76, 98, 118 |
| psycop/common/feature\_generation/loaders/non\_numerical\_coercer.py                                            |       15 |        1 |     93% |        11 |
| psycop/common/feature\_generation/loaders/raw/\_\_init\_\_.py                                                   |       10 |        0 |    100% |           |
| psycop/common/feature\_generation/loaders/raw/load\_coercion.py                                                 |      102 |       52 |     49% |33-75, 97-118, 127-136, 151, 166, 180, 194, 208, 224-233, 248-263, 277-289, 304-324, 342, 356, 370, 384, 398, 412, 426, 440, 454, 468, 482 |
| psycop/common/feature\_generation/loaders/raw/load\_demographic.py                                              |       21 |       13 |     38% |13-25, 30-43 |
| psycop/common/feature\_generation/loaders/raw/load\_diagnoses.py                                                |      197 |       74 |     62% |19, 55-98, 109, 128, 150, 169, 188, 207, 226, 245-274, 285-314, 327, 346, 365, 384, 404, 423, 442, 461, 480, 499, 518, 537, 556, 578, 597, 616, 635, 657, 676, 695, 714, 736, 755, 774, 793, 815, 834, 853, 872, 892, 911, 930, 949, 968, 987, 1009, 1028, 1047, 1066, 1085, 1105, 1124, 1143, 1163, 1182, 1201, 1220, 1240 |
| psycop/common/feature\_generation/loaders/raw/load\_ids.py                                                      |       10 |        5 |     50% | 10, 23-29 |
| psycop/common/feature\_generation/loaders/raw/load\_lab\_results.py                                             |      180 |       92 |     49% |31-54, 73-95, 113-139, 157-180, 198-238, 249, 261-308, 320, 332, 344-351, 363, 375, 387, 399, 411, 423, 435, 447, 459, 471, 483, 495, 507, 519, 531, 543, 555, 567, 579, 591, 603, 615, 627, 639, 651, 663, 675, 687, 699, 711, 723, 735, 747, 762 |
| psycop/common/feature\_generation/loaders/raw/load\_medications.py                                              |      201 |       77 |     62% |53-113, 135-144, 168, 189, 217, 254, 279, 303, 327, 351, 374, 397, 426, 452, 476, 502, 521, 540, 559, 578, 597, 617, 638, 658, 679, 698, 717, 750, 774, 797, 817, 837, 857, 876, 895, 914, 933, 952, 971, 990, 1010, 1030, 1049, 1068, 1087, 1106, 1125, 1144, 1163, 1182, 1201, 1220, 1239, 1258, 1277, 1296, 1315, 1334, 1353, 1373 |
| psycop/common/feature\_generation/loaders/raw/load\_structured\_sfi.py                                          |       74 |       44 |     41% |11, 32-59, 64, 76-92, 97-114, 119, 129, 139, 149-158, 163-172, 177-187, 192-206, 211-224, 229-239 |
| psycop/common/feature\_generation/loaders/raw/load\_t2d\_outcomes.py                                            |       19 |       11 |     42% |13-24, 29-42 |
| psycop/common/feature\_generation/loaders/raw/load\_text.py                                                     |       53 |       32 |     40% |18, 30, 75-85, 111-144, 165-181, 198, 217, 238, 263-275 |
| psycop/common/feature\_generation/loaders/raw/load\_visits.py                                                   |       77 |       46 |     40% |65-169, 178, 192-203, 212, 228, 247-258, 269, 286, 296-308 |
| psycop/common/feature\_generation/loaders/raw/sql\_load.py                                                      |       21 |       13 |     38% |     42-70 |
| psycop/common/feature\_generation/loaders/raw/utils.py                                                          |       70 |       51 |     27% |27-38, 55-77, 133-283 |
| psycop/common/feature\_generation/sequences/cohort\_definer\_to\_prediction\_times.py                           |       29 |        0 |    100% |           |
| psycop/common/feature\_generation/sequences/event\_dataframes\_to\_patient.py                                   |       67 |        0 |    100% |           |
| psycop/common/feature\_generation/sequences/patient\_loaders.py                                                 |       37 |       14 |     62% |22, 30-32, 71-73, 77-92, 96-98 |
| psycop/common/feature\_generation/sequences/utils\_for\_testing.py                                              |        5 |        0 |    100% |           |
| psycop/common/feature\_generation/text\_models/fit\_text\_models.py                                             |       10 |        1 |     90% |        34 |
| psycop/common/feature\_generation/text\_models/preprocessing.py                                                 |       21 |        5 |     76% |     70-90 |
| psycop/common/feature\_generation/text\_models/utils.py                                                         |       12 |        4 |     67% |23-24, 37-38 |
| psycop/common/feature\_generation/utils.py                                                                      |       48 |       23 |     52% |16, 37, 64, 68, 72, 93-100, 113, 133, 137, 147-162 |
| psycop/common/global\_utils/cache.py                                                                            |        8 |        1 |     88% |         9 |
| psycop/common/global\_utils/paths.py                                                                            |        6 |        0 |    100% |           |
| psycop/common/global\_utils/pickle.py                                                                           |       11 |        6 |     45% |7-10, 14-17 |
| psycop/common/global\_utils/pydantic\_basemodel.py                                                              |       18 |        1 |     94% |        25 |
| psycop/common/global\_utils/synth\_data\_generator/\_\_init\_\_.py                                              |        0 |        0 |    100% |           |
| psycop/common/global\_utils/synth\_data\_generator/synth\_col\_generators.py                                    |       63 |       21 |     67% |69-97, 126-135, 138, 148, 173, 241-255 |
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
| psycop/common/model\_training/data\_loader/data\_loader.py                                                      |       43 |        9 |     79% |43, 65, 71, 79, 103-111 |
| psycop/common/model\_training/data\_loader/utils.py                                                             |       32 |       14 |     56% | 23, 81-98 |
| psycop/common/model\_training/preprocessing/\_\_init\_\_.py                                                     |        0 |        0 |    100% |           |
| psycop/common/model\_training/preprocessing/post\_split/\_\_init\_\_.py                                         |        0 |        0 |    100% |           |
| psycop/common/model\_training/preprocessing/post\_split/create\_pipeline.py                                     |       31 |        5 |     84% |22, 60, 77-81, 115 |
| psycop/common/model\_training/preprocessing/post\_split/pipeline.py                                             |       15 |        0 |    100% |           |
| psycop/common/model\_training/preprocessing/pre\_split/\_\_init\_\_.py                                          |        0 |        0 |    100% |           |
| psycop/common/model\_training/preprocessing/pre\_split/full\_processor.py                                       |       34 |        0 |    100% |           |
| psycop/common/model\_training/preprocessing/pre\_split/processors/col\_filter.py                                |       82 |       14 |     83% |47, 65-76, 145, 174-182, 201, 217 |
| psycop/common/model\_training/preprocessing/pre\_split/processors/row\_filter.py                                |       81 |       11 |     86% |49, 85, 118, 132-137, 157, 167, 174, 183-187 |
| psycop/common/model\_training/preprocessing/pre\_split/processors/value\_cleaner.py                             |       45 |        1 |     98% |       107 |
| psycop/common/model\_training/preprocessing/pre\_split/processors/value\_transformer.py                         |       39 |       13 |     67% |40, 46-56, 65-76, 84, 87 |
| psycop/common/model\_training/tests/\_\_init\_\_.py                                                             |        0 |        0 |    100% |           |
| psycop/common/model\_training/tests/test\_data/\_\_init\_\_.py                                                  |        0 |        0 |    100% |           |
| psycop/common/model\_training/training/\_\_init\_\_.py                                                          |        0 |        0 |    100% |           |
| psycop/common/model\_training/training/model\_specs.py                                                          |       13 |        0 |    100% |           |
| psycop/common/model\_training/training/train\_and\_predict.py                                                   |      110 |       12 |     89% |105, 308-327, 382 |
| psycop/common/model\_training/training/utils.py                                                                 |       14 |        2 |     86% |    23, 33 |
| psycop/common/model\_training/training\_output/\_\_init\_\_.py                                                  |        0 |        0 |    100% |           |
| psycop/common/model\_training/training\_output/artifact\_saver/to\_disk.py                                      |       49 |        1 |     98% |        58 |
| psycop/common/model\_training/training\_output/dataclasses.py                                                   |       57 |       12 |     79% |82-85, 94-103, 106 |
| psycop/common/model\_training/training\_output/model\_evaluator.py                                              |       33 |        1 |     97% |        41 |
| psycop/common/model\_training/utils/\_\_init\_\_.py                                                             |        0 |        0 |    100% |           |
| psycop/common/model\_training/utils/col\_name\_inference.py                                                     |       41 |        9 |     78% |35, 67-68, 76, 96-101, 128 |
| psycop/common/model\_training/utils/decorators.py                                                               |       43 |        1 |     98% |        31 |
| psycop/common/model\_training/utils/utils.py                                                                    |       89 |       29 |     67% |40, 116-119, 145, 150-151, 159, 164, 213-219, 231-232, 249, 253, 283-286, 294-298, 306-316, 328-332 |
| psycop/common/test\_utils/str\_to\_df.py                                                                        |       31 |        1 |     97% |        75 |
| psycop/common/test\_utils/test\_data/model\_eval/generate\_synthetic\_dataset\_for\_eval.py                     |       55 |       39 |     29% |37, 42, 61-65, 84-88, 92-170 |
| psycop/conftest.py                                                                                              |       47 |        3 |     94% |33, 37, 104 |
| psycop/projects/forced\_admission\_inpatient/utils/feature\_name\_to\_readable.py                               |       26 |        5 |     81% | 15, 39-42 |
| psycop/projects/restraint/\_\_init\_\_.py                                                                       |        0 |        0 |    100% |           |
| psycop/projects/restraint/model\_evaluation/config.py                                                           |       43 |        0 |    100% |           |
| psycop/projects/restraint/model\_evaluation/figures/feature\_importance/shap/get\_shap\_values.py               |       73 |       41 |     44% |26-37, 42-54, 70, 78-92, 102-134, 164-172, 181-194 |
| psycop/projects/restraint/model\_evaluation/figures/feature\_importance/shap/shap\_plots.py                     |       53 |       29 |     45% |41, 49, 75-87, 96-102, 118-148 |
| psycop/projects/restraint/model\_evaluation/figures/feature\_importance/shap/shap\_table.py                     |        7 |        0 |    100% |           |
| psycop/projects/restraint/model\_evaluation/utils/feature\_name\_to\_readable.py                                |       53 |       11 |     79% |6-19, 74-82, 145 |
| psycop/projects/restraint/test/\_\_init\_\_.py                                                                  |        0 |        0 |    100% |           |
| psycop/projects/restraint/test/test\_model\_evaluation/\_\_init\_\_.py                                          |        0 |        0 |    100% |           |
| psycop/projects/restraint/test/test\_model\_evaluation/conftest.py                                              |        7 |        0 |    100% |           |
| psycop/projects/restraint/utils/best\_runs.py                                                                   |       76 |       33 |     57% |24, 30-38, 42-48, 51-53, 70-74, 77, 80, 86, 90, 94, 98-99, 105-109, 112-116, 120, 124-125, 133 |
| psycop/projects/restraint/utils/feature\_name\_to\_readable.py                                                  |       26 |       22 |     15% |6-16, 21-54, 59-64 |
| psycop/projects/t2d/paper\_outputs/config.py                                                                    |       18 |        0 |    100% |           |
| psycop/projects/t2d/paper\_outputs/model\_description/feature\_importance/shap/conftest.py                      |        7 |        0 |    100% |           |
| psycop/projects/t2d/paper\_outputs/model\_description/feature\_importance/shap/get\_shap\_values.py             |       59 |       34 |     42% |18-29, 34-46, 62, 70-84, 93-125, 153-163 |
| psycop/projects/t2d/paper\_outputs/model\_description/feature\_importance/shap/plot\_shap.py                    |       31 |        0 |    100% |           |
| psycop/projects/t2d/paper\_outputs/model\_description/feature\_importance/shap/shap\_table.py                   |        7 |        0 |    100% |           |
| psycop/projects/t2d/paper\_outputs/model\_description/performance/performance\_by\_ppr.py                       |       31 |        7 |     77% |72-84, 88-90 |
| psycop/projects/t2d/paper\_outputs/model\_description/performance/plotnine\_confusion\_matrix.py                |       12 |        0 |    100% |           |
| psycop/projects/t2d/paper\_outputs/model\_description/performance/sensitivity\_by\_time\_to\_event\_pipeline.py |       35 |       20 |     43% |48-56, 60-90, 94-100, 104-106 |
| psycop/projects/t2d/paper\_outputs/model\_description/robustness/robustness\_plot.py                            |       12 |        1 |     92% |        51 |
| psycop/projects/t2d/paper\_outputs/model\_permutation/boolean\_features.py                                      |       27 |       13 |     52% |32, 42-60, 64-73 |
| psycop/projects/t2d/paper\_outputs/model\_permutation/modified\_dataset.py                                      |       40 |       26 |     35% |18, 29, 36-43, 51-94 |
| psycop/projects/t2d/paper\_outputs/model\_permutation/only\_hba1c.py                                            |       41 |       19 |     54% |35-53, 88-113 |
| psycop/projects/t2d/utils/feature\_name\_to\_readable.py                                                        |       26 |        5 |     81% | 15, 44-47 |
| psycop/projects/t2d/utils/pipeline\_objects.py                                                                  |      108 |       51 |     53% |21-22, 30, 52, 56-64, 68-74, 77-79, 96-97, 100-104, 107, 110, 117, 130-134, 137-139, 143, 155-162, 175-182, 197-207, 218 |
|                                                                                                       **TOTAL** | **4392** | **1262** | **71%** |           |


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