# Changelog

<!--next-version-placeholder-->

## v0.134.0 (2023-11-10)

### Feature

* Support spaces within column names ([`56e1019`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/56e10193f7b46cf8a1a96f83879c375c00f58ad2))
* Add support for spaces in header titles ([`52140a1`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/52140a1c116f6a482b8576318a0d030d63a67a8b))

## v0.133.4 (2023-11-09)

### Fix

* Change BinaryClassificatinoPipeline to take sklearn Pipeline instead of Sequence[ModelStep] ([`aa5615e`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/aa5615ed548eab4a6cd38f1c6e45363e77ea4f8b))

## v0.133.3 (2023-11-09)

### Fix

* Change BinaryClassificatinoPipeline to take sklearn Pipeline instead of Sequence[ModelStep] ([`b23e6e0`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/b23e6e01b3cfcae0786100b09d77a51e720fd10c))

## v0.133.2 (2023-11-09)

### Fix

* Pass pred time uuid to binaryclassification task ([`078cb92`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/078cb92df2839b1ee90d0e0ba896aa2656a8b764))

## v0.133.1 (2023-11-08)

### Fix

* Invalid imports ([`4d27c03`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/4d27c0388b955b79c90a9d992b9839bcbab555df))

## v0.133.0 (2023-11-07)

### Feature

* Add multilogger ([#388](https://github.com/Aarhus-Psychiatry-Research/psycop-common/issues/388)) ([`7e6d8ef`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/7e6d8efe449bd3092fd2e07b9c411b3ab31295ab))

## v0.132.0 (2023-11-07)

### Feature

* Add multilogger ([`dc52e55`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/dc52e55d0eab7152dc17101244aadf5c0c600d54))

## v0.131.0 (2023-11-07)

### Feature

* Add logger.info() ([#382](https://github.com/Aarhus-Psychiatry-Research/psycop-common/issues/382)) ([`0b59e91`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/0b59e910636cccb56f243dea749b7b6396299d51))

## v0.130.0 (2023-11-07)

### Feature

* Log calculated metric with SplitTrainer ([#364](https://github.com/Aarhus-Psychiatry-Research/psycop-common/issues/364)) ([`9759f7c`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/9759f7c40d8f1c5362e1b1b5c4857d031e494bd4))

## v0.129.1 (2023-11-06)

### Fix

* Collect lazyframe and return pl.series ([`ffca056`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/ffca05615286027d03fdf07c52ac4729b61447e3))

## v0.129.0 (2023-11-03)

### Feature

* Performance by lookahead ([`671f4e9`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/671f4e9fde6d9b9d7d4c1a829de309c5ea03f1d4))
* Different lookaheads ([`d7538bf`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/d7538bf1b68309df74f25c31b9c36ebc399e7b2f))
* Add script for train test on diff lookaheads ([`41a29d8`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/41a29d8d6a43176ffafb3908f9ed7ac7100569c3))
* Add script for train test on diff lookaheads ([`ce8f59f`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/ce8f59fc2cc6bb6b7a1e2951e2ed3504a109a933))
* Add shap ([`4029cd8`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/4029cd8b0940ccbc616b3d28bec09d750027a946))

## v0.128.0 (2023-11-01)

### Feature

* Apply diagnosis mapping ([`6c75e7f`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/6c75e7f4e8df43d17021b48b6a90f541ca9fec5f))
* Add smoking and hypertension ([`a43a1bf`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/a43a1bf2f12cd929e3381a6050f039dd05593d3a))
* Added new config ([`5811509`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/58115097ea7ae1be382ee40be9948ddab9a17994))

### Fix

* Refactor tasks structure ([`12f2632`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/12f2632f16fd62ad588804ea9338114b37aed72d))
* Allow training from overtaci remote desktop ([`7e9b1d7`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/7e9b1d746a4152d66f44e0b52c015542dc97a26c))
* Remove warnings ([`66c529e`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/66c529edad11a40553a6696443be593fc55c41f2))
* Added hotfix for wandb folder during debugging ([`52fb95e`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/52fb95e180cb0bf9226639891f7a12be8b3d065f))
* Error made by pl lightning when saving hp ([`e1f507d`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/e1f507dfa21416451e62659725aa00080b99abe7))
* Added callbacks ([`983908a`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/983908a27f46714c0b7d74c9c3f1164efab00e8e))
* Removed hotfix for behrt embedder ([`7112773`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/711277318db2a1551414d86b50f6fdbd0185287d))
* Fix based on pr comments ([`4e71238`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/4e71238d9632af0bf77e0d9c8b6199a06243ec17))
* Undo edit ([`af04d12`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/af04d12af39e6ed000082dd8e25f70e62078a0fd))
* Removed todo comment ([`17c463a`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/17c463a2aa26e040e1710146910e1f82c1013d59))

## v0.127.2 (2023-11-01)

### Fix

* Allow list of data dirs for multirtun ([`d77186e`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/d77186e9c0331883ad2dd08e37a77852fd36846e))
* Update fa subset feature fns ([`7197efe`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/7197efe2e4460aecdaaa8770d81056880e73c7fd))
* Allow list of data dirs in cfg ([`064f8cb`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/064f8cb1e44abf62d2db90963cd69a495d6d1b2e))
* Remove redundant quatation marks ([`f61209f`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/f61209ff287691d7834d926efce1b61269c01dd0))

## v0.127.1 (2023-10-26)

### Fix

* Pydantic requires types to be callable. Removed subscripting of pd.Series. ([`95d89a6`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/95d89a61d8a004a90bb1137e9d3f9e900233a16e))

## v0.127.0 (2023-10-25)

### Feature

* Add cls token to behrt embedder ([`568224f`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/568224f40ecc59b7dd079b1847448b335a0d5e3f))

## v0.126.0 (2023-10-24)

### Feature

* Add blood pressure loaders ([`a9d8c26`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/a9d8c2642a9859e4995197702828dee0da39941b))
* Init loader ([`62b580c`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/62b580c2d70fbe7e60a7c58e82364da5b70079f8))

### Fix

* Error-handling ([`c5f8997`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/c5f899718cd5209aeb1eb6f50b67b50958760ef9))

## v0.125.0 (2023-10-24)

### Feature

* Add smoking data ([`6547223`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/6547223397707a8ed4bf9aa6132c66d2bc127cee))

### Documentation

* Improve documentatin ([`0fdca6f`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/0fdca6f12c528a02978b46d1200084c1f3ef9ce3))

## v0.124.0 (2023-10-23)

### Feature

* Actually use the sliced timeframes for finetuning ([`52950a5`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/52950a5c557b67e8bf3c1ddba7f8515a4c6e0836))
* Add patientslice ([`77e86bc`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/77e86bc1d89bd9321698e5a18572222e32f2982c))

### Documentation

* Add todo ([`5eab729`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/5eab729cb937bd324bd1d05bf1aab659906de034))

## v0.123.0 (2023-10-23)

### Feature

* Specify features ([`f659990`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/f6599908d845ca2db612b1a37c5bc7a97e62af86))
* Define first 3 layers ([`caeca0b`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/caeca0bb7c297d350ebb2953ae500f7a7393fd24))
* Add PAD loader ([`ccfe572`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/ccfe5721f4992d57437ca800fdcb61a466b2f583))

## v0.122.0 (2023-10-23)

### Feature

* Merge multiple feature sets ([`6526bdd`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/6526bddb974db3af8bcadccbfac6f975f4d2ff25))
* Test xgboost assumption ([`5a3222a`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/5a3222a5eed3bbf1d8519d4e8ee1f4e940c3f263))
* Add test of xgboost hyperparams assumption ([`0768f62`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/0768f62150c954512950b55a4803a6bbd4533c1f))

### Fix

* Misc ([`b63d22d`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/b63d22d0430a7d1babb74c87f5e5eef6524106a4))
* Misc ([`f8e35a0`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/f8e35a07a20eea2645b1d5c5b649d58e42a2bada))
* Correct checking whether dfs can be joined ([`7630572`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/7630572897974cd53c9ee2ceb0aefa0e64268967))
* Move feature merging to data_loader ([`a31c9ca`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/a31c9ca02838a04c759b45566d6f347e80798ad9))

## v0.121.0 (2023-10-23)

### Feature

* Add devcontainer.json ([`b9230b4`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/b9230b4b27b1ff337e1c121846c3d76fd1e3ef46))
* Allow levels of granularity in diagnosis mapping ([`a06fd75`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/a06fd75d6670574e5f4fe4a13fff1beb191d3a48))
* Add subsetting script ([`dcd10ee`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/dcd10eefef52d8296aafcea4d9725e83c89f1cc8))

### Fix

* Update train val descriptive comp script ([`b1f3e72`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/b1f3e72fa2a2546176e3034ca99b15467d738657))

### Documentation

* Comment test ([`7dbeb53`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/7dbeb53b3930986d7ee70ad5d547ee1ed9773700))

## v0.120.0 (2023-10-18)

### Feature

* Extract runs to functions, to avoid instantiation on import ([`afc94cb`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/afc94cb29fc9a2995f7003efd62638ca7afec6d0))

### Fix

* Renames ([`5ea1fe5`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/5ea1fe53b3ff2208885f10cc168858b2ddf66936))

### Documentation

* How to install cuda enabled pytorch on overtaci ([`25608d2`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/25608d27fafdd3f61862479d6d177aed478268d8))

## v0.119.0 (2023-10-18)

### Feature

* Create plot when training xgboost hba1c only ([`cd52ec8`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/cd52ec89c18967453cb387d47ad3346a8baea431))

### Fix

* Change typehint for patient colnames ([`22d9317`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/22d931731fc96d42e8ced0671e63a44be9dcfad5))
* Do not import get_best_eval_pipeline unless main ([`d5da51f`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/d5da51fbf9c66e1ef5c75d42657f1002b56a3605))
* Fixed mutable default error in config ([`a2d8294`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/a2d8294f62c4b19a6b2801ce9a9898585deb4327))
* Source subtype filtering works ([`1259203`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/1259203d58b09ee2b09f71f7455976199fc9ed10))

## v0.118.0 (2023-10-12)

### Feature

* Add overwrite eval warnings ([`6d5657f`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/6d5657f52becfa6a40e1ee6c958dffc71808eb36))

## v0.117.1 (2023-10-12)

### Fix

* Update naming in cfg schema ([`5a1f131`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/5a1f1319abf9bd5e85adbe577981a3cbefcbf31e))
* Wrap eval function ([`b2f5161`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/b2f5161301f1cc69c14e575d7a2f9db9d44bc1aa))

## v0.117.0 (2023-10-12)

### Feature

* Added fine-tuning script ([`bef7c88`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/bef7c8859fe5735cccd5be267f59be17abedb6fb))

### Fix

* Ran precommit ([`6e6bf80`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/6e6bf80b1ae9cc972f3a45577140deb64c983136))
* Don't run bf16 on tests ([`5a624ca`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/5a624ca8ba76e5c59f3b1c6799fbfe8633540264))
* Ran pre-commit ([`11536be`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/11536beab426a3b99d5361ccca77b3a835ef5f80))
* Ran pre-commit ([`42b8e29`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/42b8e29bb22c78d06eacd13a2453d3e40b9fb14d))
* Added description of how to create checkpoint ([`453c5a9`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/453c5a94580077bd137c712edc6042b68addc469))
* Updated the checkpoint ([`b04f37c`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/b04f37c33bfab46e492bd0544b57d2f4fb558ff0))
* Ran pre-commit ([`7e40ed1`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/7e40ed105f18c2320601ad120a34b4ea64fa3e8a))
* Added test for multilabel ([`1f29927`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/1f29927f2be9995a2e473dc3419f41bfce6d20d8))
* Based on review from @MartinBernstorff ([`643c1e4`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/643c1e4692c96ac382a505214cb0e568c5612b0b))
* Ran pre-commit ([`b2a98ec`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/b2a98ec2d1f19379be8b45bab27e3beecad24528))
* Remove .conftest antipattern ([`5e5a693`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/5e5a6930c858ac5cb75fa2466d8c756f19439614))
* Ran pre-commit ([`fab110a`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/fab110a9d2a32c4ac5daad56bf00552cbd65e566))
* Ran pre-commit ([`ff41ad2`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/ff41ad2a7939952c8502d926c7a4a4e431b430cd))

## v0.116.0 (2023-10-11)

### Feature

* Filter diagnosis subtype in BEHRT ([`2c56baf`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/2c56baf1baf2b0f2285d8b24c0f23d7aee27f96c))
* Generate pred timestamps without washout ([`911076b`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/911076bc7d35956f780db23e01a2a92632c0b322))

### Documentation

* Added test documentation ([`448ce4d`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/448ce4d180e16b27f291dccab8865d50234c141d))

## v0.115.0 (2023-10-10)

### Feature

* Add tasks.json ([`84546d1`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/84546d1d1b1f668f797728db435c716c13279af4))
* Add vscode dev task ([`8a349b8`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/8a349b8ce7b4b1978897ab0245ae34d99e0dd1e7))
* Create diagnosis mapping (icd10->caliber) ([`e27af96`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/e27af96a0f8422b858e514c1a423daf529e04b59))

### Fix

* Delete unused file ([`1e4cab7`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/1e4cab7fea33bc455b4b4c1cd7289d14f03e90b1))

## v0.114.0 (2023-10-09)

### Feature

* Add procedure codes ([`002d488`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/002d488411902bd3f88b17831dca2c8826b445ed))

## v0.113.0 (2023-10-06)

### Feature

* Define cohort ([`7434bac`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/7434bacf131f387c422d1b2748acd82ac209a00f))

### Fix

* Adapt ([`d0977ba`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/d0977baac53f2186b813b679d99c73f22637a568))
* Minor changes ([`9969d99`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/9969d99432f6da55f18f80c0bd916d06e304ce2d))

## v0.112.0 (2023-10-03)

### Feature

* Gradient accumulation fix OutOfMemoryError? ([`7cfc47b`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/7cfc47ba47c8a92b6d5f663cc4cae275fdf7947f))
* Lr scheduler linear with warm-up ([`0f2d433`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/0f2d4333cb8e8722cfa725bf8c5bf100835e0af5))
* Pretrain version ([`5e9091c`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/5e9091cc85ff26d8334d27215dcc001109642170))
* Ready for training ([`1218794`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/1218794ffc20c1f5ed53bbcfdb4f4affd0485d6e))
* Expand test to cover model checkpointing ([`1f0bece`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/1f0bececb1d2b5ec701b205538716133fe6778ad))
* Lightning module saves hyperparams ([`59e7ac6`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/59e7ac660810848fc79cd27df19add19bd4d65ca))
* Adapt sequence training script to pytorch lightnign ([`005cbdf`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/005cbdfce8366021e0f4f11ad02b5f8e7a3843ce))
* Initial changes to pytorch lightning module ([`d919009`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/d919009d5323bdce64cb6ba7715bcf16cae10f90))
* Added training script for sequence model ([`7d2c2bb`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/7d2c2bbd41db933ad7d139ada551654ac88dea0b))

### Fix

* Configs should be initialised with factories ([`7ddbf9b`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/7ddbf9b8908010c79ff94bbf03b13db38cb37588))
* Ruff ([`626e0fc`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/626e0fc75f6a203a04838a964d958c806b37f103))
* Fully transitioned to pl ([`d3a8d32`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/d3a8d32f74f5e35a774b32f56ccfaa52f8e1b493))
* Replaced print with logging statements ([`769530f`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/769530f036c03f801d5b44bac109ee7b5305a4e2))
* Make sure parameters is actually moved to the gpu ([`131e1f0`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/131e1f01610f5fe0aef360edc7ad94d4866ee1ba))

### Documentation

* Typo ([`0dfeb69`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/0dfeb69acd4c0a834bf8d793c0fb541f728615b6))

## v0.111.0 (2023-10-03)

### Feature

* Add dev container ([`84301b8`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/84301b82020fffabd9573e70bb9ada6ad3e8c7a6))
* Add corr plot ([`d0c116f`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/d0c116f16eb6b8f2fcd31f10a29b5a3cd1ca6602))
* Add feature outcome corrs ([`4376b19`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/4376b190f2aee4350888f8c7d9908c987a7cc3f8))
* Add hist ([`708ca45`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/708ca45b8beb67ca091b9d08fe1efe93c41bc5bd))
* Descriptive stats ([`8c0b038`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/8c0b038f8ccb82090e1b4b24308fab70b1b2d02b))

### Fix

* Changes ([`d172358`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/d1723589ad21b5da4ca5f3d40794f5c26cbbadd1))
* Minor errors ([`1b21da3`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/1b21da37e234603e29cffbccfb34eac284b7984a))

## v0.110.0 (2023-09-26)

### Feature

* Time from first pos pred to next hba1c ([`b0d805d`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/b0d805de4a82b84d452ef7fc550c8d887bcda290))
* Ned script for retraining model with new cv ([`eba749f`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/eba749f8c82f74b3c3997362e5ab68ce4b5c6343))
* Add feature importance table ([`423cc23`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/423cc238450f8095d6a10dde5ced100dfdd455e4))
* Add baseline table one ([`8695e29`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/8695e294f687b40b0a12eafe832ab56339bb5259))
* Adding eval plots ([`5b81f26`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/5b81f260309759e24c10b61aa26d4114025d85ae))
* Add new eval branch ([`5e8aea3`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/5e8aea3477a5418a16f7791885ff0474f53f8621))
* Wip new eval structure ([`8728545`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/8728545f36ff86cb48a38537a7986d6354bc0baa))

### Fix

* Lint ([`5bbdd7c`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/5bbdd7c06527865f88d1b5a6c43cef2e685ee0f5))
* Missing path arg ([`f4fe5e1`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/f4fe5e1e7b73be47652a5952e126b1341611899d))
* Spelling in comments ([`58de461`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/58de461ae9e9ec391c054def7b2110d31ca0299d))
* Lint ([`0eeb957`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/0eeb957156dee907fcef7ec3fc36f49e76e35d8d))
* Lint ([`f084a83`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/f084a8365d60958982df9c0a22263fba7722c2c0))
* Various minor changes ([`4a8f6de`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/4a8f6de407ce6f12ecd2d670e4a2be4c86b65712))
* Missing return type annotation ([`cd2f8f9`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/cd2f8f945fe6c2fed05b8dcab3a360404f72b2ae))
* Formatting ([`b8521e4`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/b8521e4791ab01cb93816076a588490002cad0c9))
* Delete old eval folder ([`5e5cc6a`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/5e5cc6a8cf5c52e1c1d0ceef8fa98b886c830602))
* Eval paths ([`6f1f1a1`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/6f1f1a1231134b7a3ea32fd235787d7040ae66b6))
* Selected runs ([`a448ab4`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/a448ab427a88510dd3d73da7644098795adabf84))
* Configs ([`80d1907`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/80d19070a5f6c0a0db021f933a477e947edbe896))

## v0.109.0 (2023-09-14)

### Feature

* Implement classifierchain ([`097cc0a`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/097cc0a85db563e01a226fa0b48ae0039e4295c8))

### Fix

* Unpack dataframe to series in eval df ([`dcf9b93`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/dcf9b93df1a2ff232733824b9728f64760008b80))

## v0.108.0 (2023-09-12)

### Feature

* Main test passes 🥳 ([`622adf7`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/622adf7b8a76c74c1b5c89b97a8ba270d79f761f))
* Add missing methods from PSYCOPModule to BEHRTForMaskedLM ([`79f8a26`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/79f8a26fb8629c43521400a847c1031faa044156))
* Update trainer to match checkpoint savers ([`ab9a234`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/ab9a23475d1ab82654df06ca0ff671dfdaf25ecc))
* Add wandb logger ([`7eef76a`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/7eef76a475d28c5809da29585e0a18cbd8560752))
* Flesh out trainining ([`40b1032`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/40b1032c16fcd62092ba6a600021d9105ea3c73d))
* Add dataclass-based vocab ([`25f7d3e`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/25f7d3e607b1a804ce40b030b7d9654365556e63))
* Implemented masking task ([`b0ffbf4`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/b0ffbf418bf0b44cea89c9eb82bb7ad8496d3485))
* Embedder skeleton ([`81479c7`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/81479c7e75d39ba7b724a58e6c9c70d55fbcc6e3))

### Fix

* Fix error from static type checks ([`13f5a38`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/13f5a3841d7371be25e2743865d62e985b837f15))
* Updated format of the mask function to allow for testing ([`d49dadc`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/d49dadc6043c1f46daf941b31be8ae38efd401ff))
* Make sure that the tests test the outer masking_fn ([`679cd54`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/679cd54c10fe5bb1ec4afad6dd8f24f42da98934))
* Renamed PsycopModule -> TrainableModule ([`fadc9a1`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/fadc9a12b2ebbbe7187fd419725e33a7fbaaac5f))
* Remove testing assumption from Logger ([`861c162`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/861c162782b17bdc0b1bed552fc4b893fd59d32a))
* Updated logger to handle allow logging configs seperately ([`604dc2e`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/604dc2e3b14874e9e0e14e946d60558f470634aa))
* Moved logger interface to its own script ([`4c83ff4`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/4c83ff40aebd299cac03fc0992f5b412390da0fc))
* Added vocab_size ([`3789834`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/3789834edd30dec5a53f0ca36fdf761d7e324080))
* Added type hints ([`370aebe`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/370aebecca6512326678ab21a784141878854069))
* Forward pass in embedding module works ([`5cd38d9`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/5cd38d91d935598b994dd11aaa374f78f86e883a))
* Added patient dataset ([`2a00c32`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/2a00c32891fc5304b48256a3943161c0c6e9a097))
* Added behrt embedder ([`a2bbd8b`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/a2bbd8bd534afffa264c753a863e41b9e79af720))

### Documentation

* Removed old comments ([`74ca7d3`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/74ca7d38bfe59a81cb53a716c53a9c6e493fb1ed))
* Removed unnecessary comment ([`6f8b175`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/6f8b175ba448c5ea904b7926c45b41dc941eed23))

## v0.107.0 (2023-09-04)

### Feature

* Rename cohort definition to cvd_definition ([`27c5e59`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/27c5e59f0acd807f41cf6f2e99674a48d49f7356))
* Minor examples ([`f2d99ad`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/f2d99ad00c2c2301540693f56eb20a33a52e393a))
* Cvd outcome definition ([`89dcf49`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/89dcf4948ec2b81012cfa44b0582d78d96c8f820))
* Add cvd filters ([`9dcfe33`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/9dcfe33439a52199e4764fc0d49cb76596d9fe1d))

### Fix

* Remove use of hba1c in cvd filters ([`395487e`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/395487e481154eb0a3048ef6547685909c72ac48))
* Unneeded newline handling ([`311d15d`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/311d15d712d8fb79377f927f9cb61382a0b651eb))
* Strip lines of whitespace before generating dataframes ([`628db49`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/628db494dd0753b69b69c63797b37a0ae76f33f8))

## v0.106.0 (2023-08-31)

### Feature

* First version ([`2fc715c`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/2fc715c099c94b9dd470595f73546aa3b9c0786b))

### Fix

* Possibly unbound variable ([`864f59b`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/864f59bfa20d27e5b8ec28a5b9d84a3a2ac487ba))

### Documentation

* Point to patient object tests ([`d727664`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/d7276642d5a650110a6693c6428058fa3784f432))

## v0.105.0 (2023-08-30)

### Feature

* Add tfidf ([`fba845a`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/fba845a9dc2fb431da5637a4348c3413cce4797e))

### Fix

* Config of last model ([`250d854`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/250d854cb8b6e29260e7fe0fb359a1463e18c7ae))
* Naming ([`5bca53b`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/5bca53b1c993bd15c97a4944a855dbdc88e51a37))
* Configurations for new tfidf feat set ([`fe33f0f`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/fe33f0fbe6b4a2ae7b3a4894b7aaf30960d5eb8a))
* Update configurations of model train and eval ([`dd0f83c`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/dd0f83c9e632904b87c2d0cfe4ee94653b8a2474))
* Reconfigure text lookbehinds ([`f6151ad`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/f6151adf3d9e9b25d09451f27e8292adb9e2188b))
* Text specs ([`a0ac8bf`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/a0ac8bf30f3214e2645a36261da56dabb2a8746e))

## v0.104.0 (2023-08-24)

### Feature

* Parse date of birth to all patients ([`c469997`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/c469997d39f3ee42b5a3aad06d0ba54574c4fcc9))

## v0.103.0 (2023-08-23)

### Feature

* Train new tfidf model and encode text ([`1fdd3ce`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/1fdd3ce7fa2534f00f0cdd997443d6f2624e5621))

## v0.102.1 (2023-08-23)

### Fix

* Don't shadow python builtin ([`c05d307`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/c05d307d70ef00433405ec5188ddeb5a1ce7110b))

## v0.102.0 (2023-08-23)

### Feature

* Get patients from sql ([`d9ebba4`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/d9ebba418e3e777776d02a1b4dcfe4e5d809cdf6))

### Fix

* Rename from merge ([`38257e4`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/38257e442fe3aa31a3c1a29b92eac02931c83897))
* Type checking block for circular imports ([`a33a8de`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/a33a8de19f0810818b093029954ddd66dac75b46))
* Typo in shak codes ([`e2cd184`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/e2cd1846f55c930f61cad2afed5c41b61e263deb))

## v0.101.0 (2023-08-23)

### Feature

* Convert getters to properties ([`cbe130b`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/cbe130b191bd3047aee0295fc3594129674d4341))
* Handle lookahead-based outcome resolution ([`489003f`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/489003f8544b2c76c5d30cb788c148bd1e9f02c0))
* Remove patient_ids and fix downstream type consequences ([`2918452`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/29184521b5f3c6721043939ce69bf5fbadfb8aa9))
* Misc. ([`bd3d6ea`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/bd3d6eafcf9ddcf896f130cecc4bb059bf8e20e2))
* First working unpacker ([`c65219b`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/c65219bb984d0ea1202c924e65220f50d9207b2b))
* First stab at unpacking to patient dfs ([`74ba12a`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/74ba12ac405f3c7466f16c2b743e4e9cc37c6b0e))
* Filter prediction sequences ([`402eee9`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/402eee9d14f17a5d77e7a11439d0cd816c30ff8c))

### Fix

* Rename patient id in tests ([`da79654`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/da79654cae87541ad9f0325b17559a0f7d04a847))
* Spelling errors ([`114b624`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/114b624918b9cdff515d84c4ba8d991356b63b0d))
* Missing type import ([`575689b`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/575689be2990ecac79b97da99efa06526ba3fa1e))
* Downstream type fixes ([`1cd438b`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/1cd438b51f9a3de46ae38d2fd3e0463714e8f20b))

### Documentation

* Add comments explaining __eq__ ([`d99c0be`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/d99c0becfe4fbce05b0c599c233fdd58d178e785))

## v0.100.0 (2023-08-22)

### Feature

* Filename check earlier for feature-gen ([`17b3404`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/17b3404a3721167f6efad037855655bed8b45831))

## v0.99.0 (2023-08-22)

### Feature

* Cohort creation for the cancer project ([`4a408b9`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/4a408b9f3ccbea0d28acb980a400d5e6b0d8890e))

### Fix

* Correct type hints for aggregation ([`ac9fc29`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/ac9fc29172cca3081fe76f9e640c3689f086a9d7))
* Reconfigure lab tests ([`caebc33`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/caebc338019f0f05123f38453249ba33a6fb8856))
* Replaced unsued function ([`75f241c`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/75f241cafff64b55eace870ed73791982d2baa3f))

### Documentation

* Docstring ([`e7a6879`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/e7a6879cf48fcc0ea0d10d2663dfac1e9a78ae2d))
* Docstring update ([`c98bd22`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/c98bd22ce279d7dab440918772b0312f83612bd4))
* Update docstring ([`87c5c89`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/87c5c89bcf9cad637e37d4059f63be4707e3d039))

## v0.98.0 (2023-08-22)

### Feature

* Add sentence transformers features ([`8a91048`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/8a9104865f68da1fcf74af37ba0f338e8a3d5b9d))
* Adding text specs ([`27b37ac`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/27b37ac22c9dddbaa422c67a20afa21610d77373))

### Fix

* Type annotation ([`a18cc64`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/a18cc64996afa1fac283ac1b3388aebd1b54ff34))
* Wandb back to offline ([`b67d4f7`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/b67d4f7bc33ec4cf2f0fe1f24c804707816a8c85))
* Config ([`f5e2c73`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/f5e2c738c2fc8bbd592ec1f90902786a4adfbccf))
* Wandb config ([`761ea6a`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/761ea6a16b5a333c0c7491035323f2653edf10ae))
* Lookbehind combi config ([`09919f4`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/09919f470efe681a8c305ef0e30f90e8fe549bbc))
* Update configs ([`6ad6fef`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/6ad6fef3f8b1d8b258c15d4a165c942612e4593a))
* Correct type hints for aggregation ([`d6f4311`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/d6f4311a34b34a0deebd8ce2d10bc0bd683945cf))
* Param changes ([`650912d`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/650912df0ea1e2ef276e01fc2da1187696981417))
* Add missing arg ([`500be32`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/500be3288b0f382403f71f924bd0ad3698ce3645))
* Minor changes to params ([`dd1dcea`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/dd1dcea57e89baee88390ddaa37b7c17071de74b))

## v0.97.0 (2023-08-15)

### Feature

* Add new dir param and user prompt ([`fe46dbf`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/fe46dbfc528fc83e00fdb371a70d2436c3fb4278))

### Fix

* Broken tests due to missing arg ([`74cd065`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/74cd0651711f0292b11c9dfe84f292e86bd3a119))
* Add arg to general function ([`6395109`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/639510928f28221c8c0f0068efb90846b0c49a4d))
* Update general function ([`81506ee`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/81506ee857188b1df32ac785b12d422f42701732))
* Instructions in README.md ([`c2de501`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/c2de50141da8bb0a6e85fe522ec8e2cb66687687))

## v0.96.0 (2023-08-11)

### Feature

* Encode tfidf values ([`14806f8`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/14806f8ddaa6bccc1598ea0d4a3f559cc7f0b326))
* Embed text with tfidf model ([`d3c8c25`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/d3c8c255b4b0ce6691337c9e06ce4a36ccf98f48))

### Fix

* Ignore type check ([`17a3b45`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/17a3b45052dde0f5b6e94645d7c87b15d743ad0a))

## v0.95.0 (2023-08-10)

### Feature

* First stab at chunked feature gen ([`0a31a61`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/0a31a613282d87927d4117992c6b9f46e7ee9c2c))
* Add loader for embedded text ([`40c8271`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/40c8271bd7b45d5c9fbb68b661fd21f385a0aad3))
* Train sentence transformer code ([`055a572`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/055a572e0151b6044db9a12f491b8d308307b281))
* Sentence transformer embedding ready to train ([`40ce08a`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/40ce08ab8baa429a7529506938b02015dcf7c50d))
* Sentence transformer embedding ([`9b998f8`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/9b998f8cfd71e79a33cdd08024a45314923ee345))
* Vis qc ([`ec8ff15`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/ec8ff15dc22524137d2a0f4f753377c288a172c5))

### Fix

* Misc ([`df49d9b`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/df49d9b3772daaf5017c560abfbdb267da4c8e22))
* Ignore old import erros ([`d6a2105`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/d6a210572a4d48abb8bfec4396f0668dee19e2f9))
* Reinstate 'prefixes_to_describe' param ([`761c6f0`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/761c6f06ca34ec3b3f8ea7ee13f522285974e158))
* Remove old param ([`4240440`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/424044086c361391fe2d93c9e6c524df466843e2))
* Minor changes and typos ([`2341abd`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/2341abdec245c7558a961d53b3ba2a98374a2bbd))
* Typo in requirements ([`d97aefe`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/d97aefec0c37af65e26903313c274af091df2ee7))
* Text feat specs resolve mltp to mean ([`9dd1065`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/9dd1065e10d22a5f5db9eda2d1527b77549274f9))
* Paths ([`b008a95`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/b008a955c6d900ac089b6b85e9ca9295e86c16a6))
* Change chunking pipeline ([`781d442`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/781d44229c450eabfcfc462a34ac5a6a6eac5c98))
* Updating scz_bp feature gen ([`8b1ee14`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/8b1ee142693cdf15faaa0540fba878f0cc894d7e))
* Move chunk tests ([`df7834d`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/df7834da0494684d0da6eaae9d15df243cd9c1da))
* Move chunk tests ([`295e4c0`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/295e4c045c06375ebe464db376bc6607bfcbec65))
* Don't modify prediction_times_df in PredictionTimeFilterer ([`e1eae8c`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/e1eae8c96368d69346feb2cbccf6891af595e536))
* Type hint for ColNames ([`0454b42`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/0454b4265144b6749da2d35bfa8d144484ae1710))
* Chunked feature gen ([`d8e2ac7`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/d8e2ac772ac583600d82692dfca56e73a895cfac))
* Set wandb to offline during feature gen ([`5ce32be`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/5ce32beea252c68dac9b6d7e454f1bc57141ef5e))
* Print time taken for sentence embedding ([`ac4e8a2`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/ac4e8a2a76cd87a70db3677055d24236188e8106))

## v0.94.0 (2023-08-03)

### Feature

* Multilabel classification ([`5285b2c`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/5285b2c00f63a74cb19b6051903fd85a0df75773))

## v0.93.0 (2023-07-20)

### Feature

* Simple qc of text ([`eff9bfa`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/eff9bfa673cd0955797531d366df76dd783511fa))

### Fix

* Misc minor changes to qc ([`efc45e0`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/efc45e082bab5d7c02e3cc9f5da0daed64b30cb1))
* Working qc ([`add0333`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/add03330109e1eee94ef825e6e8a3b6d2a1a272c))

## v0.92.0 (2023-07-18)

### Feature

* To polars|pandas method for EvalDataset + fixed threshold ([`52fe185`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/52fe185099e4d1478759a01324bf55b17684b15c))
* Add loader for first visit to psychiatry ([`9ee9727`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/9ee97276624d144a1d5b53ab8aa2d9fd8ff2e018))

### Fix

* Set pythonpath for interactive session ([`ded552d`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/ded552d6d44a08869df3097212485725b0863f44))

## v0.91.0 (2023-07-18)

### Feature

* Change prefix for supplementary outputs ([`9b2a15d`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/9b2a15d9408d88d8c54fa876284ac6f3acb7d709))

## v0.90.0 (2023-07-11)

### Feature

* Only print failed checks if there are any ([`ba645ec`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/ba645ec44f52937a4575876980fe5b82b32818aa))
* Only do feature description of columns matching prefix ([`dc45ab7`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/dc45ab754e82357838b19ba2171ffb58c9e88ca7))

## v0.89.1 (2023-07-07)

### Fix

* Add birthdays as default ([`17bdc01`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/17bdc0165117d03d9cda184c74a767de6673919d))

## v0.89.0 (2023-07-06)

### Feature

* Add loader for therapeutic leave ([`a9f95ec`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/a9f95ec98fd9b8eedc3cc2fc40040aa444547908))

## v0.88.0 (2023-07-03)

### Feature

* Freeze DataframeBundles ([`8a1f29b`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/8a1f29be5e5e6e7edfd3e76d9da23f15b99293b7))
* First stab at types and tests for sequence windower ([`a68afd7`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/a68afd7c67efe583216dcc5452d8379935acc8aa))

### Fix

* Correct types for aggregation funcs in t2d specify features ([`6796f0a`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/6796f0ae7e2bffe7b39b4dee644e4f4f261bfe41))

### Documentation

* Add docs to eventcolumns ([`e6585a2`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/e6585a2c4873d1a1d1f98786d47c562d7dd94bd7))
* Explain sequence columns ([`be1f89d`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/be1f89ded0cb0a8d6aa2ceb5cf3a50b9ecc5d418))
* Define behaviour if lookbehind is none ([`26a7f0c`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/26a7f0c9bd15cb6c4b0a418a2a3ef2a88e244694))
* Add docstring ([`fe5fd2e`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/fe5fd2e9897619725332a24a738ca2717a8a62ce))

## v0.87.0 (2023-06-28)

### Feature

* Add INP and TCH rules ([`4a3d02a`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/4a3d02a37c8cb0eb0dcafb793c458d799d82f243))

## v0.86.0 (2023-06-28)

### Feature

* Misc from review (wip) ([`62342b3`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/62342b305e7cde6df09dcdc73900d7cb82dbc8de))
* Add cohort abc ([`387aa6e`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/387aa6e93825aab3fb1371167a104be562ac69e9))
* Outcome def for scz_bp ([`79d057d`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/79d057d85906c150ee9fa640f38456f6d71e4cc7))

### Fix

* Misc review ([`1c2ed6b`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/1c2ed6bcbaa979fe2e8d07973201faa9e4d96144))
* Create wandb folder the right place ([`081c8c0`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/081c8c01e71b4cbf51d52ee63f1f0339fcfeac5c))

## v0.85.1 (2023-06-27)

### Fix

* Remove duplicate csv ([`10b98f7`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/10b98f7355062ee35c1cf133dc01acd5538bb972))
* Merge over correct fa eval files ([`879e166`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/879e166d58e2eef872eca264741e53ffaedbd743))

## v0.85.0 (2023-06-20)

### Feature

* Add plot code ([`8c0b347`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/8c0b3477051a0e4c07481b1d98e0d598316b2c01))
* Remove name and build-system to avoid pip install -e . ([`63e871b`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/63e871bd4f1a1b5fd2895905f2bbcef119a5fb35))
* Migrate to requirements.txt ([`cab38a5`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/cab38a5ba2c42b3037ef306fa3afcb97937070ff))

### Fix

* Reset project seed ([`8cb0677`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/8cb0677ea7a11bb31284cea5ea9450b44a8df74b))
* Order in plot ([`6974eab`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/6974eab20eed301f5e37f19a150857482e952d32))

### Documentation

* Update readme ([`1214fb4`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/1214fb4ef3fc038c8657670be6914ac49772e3b5))
* Match docs ([`27d9661`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/27d96616978864224f8c8a7063234006cad71dbd))

## v0.84.0 (2023-06-20)

### Feature

* Add print statement ([`7924dcd`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/7924dcded8d514b1407b75e8844663f45d9f837e))
* Adding function ([`a08fdac`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/a08fdac8687d9712d988a7590e417ba5379a925b))

### Fix

* Add correct new models ([`fe9d4bb`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/fe9d4bb9285546cf8c8be6e54a1947e06351d41a))
* New best models ([`2db0f03`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/2db0f0385eaeceada59693147de9358d6999011c))
* Adding more flexibility ([`014ba4e`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/014ba4e6a25988432b2828649db44a76ad6c44df))
* Adding more flexibility ([`fca7264`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/fca726429a3c4447185ee644ac25cfba97710b3a))

### Documentation

* Update figure terminology ([`d0e8cc8`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/d0e8cc8ac6e1f0ecdee5ea3c9225a40067ed723c))
* Update docstring ([`82498cc`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/82498cc7e8a8c75fc9cde4fedc795b3b7b6afa9c))
* Adding docstring ([`0b8e5b4`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/0b8e5b41bdb4aac79c6747f0c2b026c594418e2a))

## v0.83.0 (2023-06-19)

### Feature

* Eval pipeline works ([`a663436`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/a663436072abb38f06f12d974de129533b398bf2))
* Add typehints to feature specs ([`bf42de0`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/bf42de0ec820607c891aa737bf39fb3b4bab0444))
* Turn wandb off for now in main feature_gen script ([`2ea9ef3`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/2ea9ef36f8b067c36033d334dc32bd0fd345ec8b))
* Cancer project initial setup ([`de8f9cf`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/de8f9cf4d713c05ae5e82821af3f7a810dfe79e8))

### Fix

* Minor change ([`41e07b2`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/41e07b288480bada959f0537148cc9238b352740))
* Update readable feature names ([`a4073c3`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/a4073c358fd5c3cc21b6525225d03c397da801a6))
* Update readable feature names ([`114cd24`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/114cd247d19d8b9c65986ab079dde9ac902d7a44))

## v0.82.0 (2023-06-14)
### Feature
* Add careml to monorepo ([`534400a`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/534400afb70b5d3c2b662e3b0bdd2350067be1d4))

## v0.81.0 (2023-06-13)
### Feature
* Move markdown handling to common ([`bdfeafa`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/bdfeafac5f55490233f17988b006ec6a1f027307))

## v0.80.0 (2023-06-07)
### Feature
* Misc. ([`482db66`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/482db66238e0276df827af39284ca2cf41000353))

### Fix
* Guard for newly optional configs ([`e9ff39e`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/e9ff39ef52c3ee3204f787478331bd4f314c147a))

## v0.79.1 (2023-06-06)
### Fix
* Remove project specific md code ([`945a0fd`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/945a0fd0597000de2197df502e983f75a804b948))

## v0.79.0 (2023-06-02)
### Feature
* Simplify feature describer ([`a9f9f7b`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/a9f9f7b9906559f61867eed90d347bbd55620b72))

## v0.78.1 (2023-06-01)
### Fix
* Patchwork grid of size 1 ([`b159a10`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/b159a108b896511d540c44dfcfbb89d9cb33b361))

## v0.78.0 (2023-06-01)
### Feature
* Increase size of axis labels in t2d pn theme ([`71b8dd0`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/71b8dd04e56288244188dd7fbaa017740adb7cd1))
* Increase size of patchwork subpanel labels ([`384e06d`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/384e06d829075404e48325595b5fed37a5414fae))
* Make HbA1c only configurable ([`d2854a8`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/d2854a8d2695900a923260c98eea82972a19c005))
* Adopt boolean dataset to featuremodifier ([`5188047`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/51880477c34ae803db51a5500637e45b35f2decf))
* Ignore static type checks on Ovartaci ([`840c015`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/840c015f06b3eb315f6c024518657c59b4ed853a))
* Allow disabling of column name checks ([`ad519be`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/ad519be2bfd16f6d927cd99ffcd68a7122468945))
* Boolean cols in place ([`8b968e5`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/8b968e543010b05de7fd11b23efcb514b398b9e3))
* Use native polars column selection ([`ef25f17`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/ef25f17571d96721722c04fd59c5111a915c6b2c))

### Fix
* Imports ([`e31e18d`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/e31e18df498be78b35e57c5b2edede2c8970d956))

### Documentation
* Improve docs wording ([`173807e`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/173807e46001f789b371824cc6ada49a980279e1))

## v0.77.1 (2023-05-30)
### Fix
* Correct lookbehind selection ([`3807a94`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/3807a9468f1225d77134786298a88d82185b71c8))

## v0.77.0 (2023-05-26)
### Feature
* Implement full supplementary generation ([`530d972`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/530d972926a3371fa95cfb55778ecf38f38fea82))
* Switch to TDD for md_object generation ([`35b4787`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/35b4787be204a4d48cb138dccd5e05de148f3c6a))
* Create required wandb folder when initialising wandb in WandbHandler ([`41037d9`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/41037d9c35de294c0116a7697b4754b8861cb8ef))
* Misc. ([`0a54195`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/0a541954135fbc5f91cb78635fb967ecdb3b1976))
* Eval run on test_set ([`76644ee`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/76644eee75f2f3b58c9538f701e17bc37f1e6250))

### Fix
* Align plot and table for median warning days ([`bda3eed`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/bda3eedc117c906bfa1b2f7f7c59f683fe7420ca))

## v0.76.0 (2023-05-24)
### Feature
* Automatic robustness figure ([`07f9f2c`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/07f9f2cada678d19e85fef6e139c029ae8a41c5d))
* Abstract robustness plots ([`514226b`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/514226ba12b61ef6bee9ba8a93adf8e907dfbedb))

### Fix
* Ensure X_by_group returns a dataframe ([`57b3160`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/57b31608278bf58be7da5227511e65fafe067162))
* Ensure X_by_group returns a dataframe ([`efab826`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/efab826354785cee8e18d7f761f70c0e4731a263))

## v0.75.1 (2023-05-24)
### Fix
* Pin wandb version to avoid failing on tests ([`2a92dda`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/2a92dda845db5a49264454b20843eb86733ae5c8))

## v0.75.0 (2023-05-23)
### Feature
* Generate a publication-ready performance_by_ppr table ([`32c20ed`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/32c20edfa1cfdd855c505e546b6304bbf1ffa2a9))

## v0.74.0 (2023-05-23)
### Feature
* Add thousand separator to conf matrix ([`fc9b6dc`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/fc9b6dcc980cbcbc51af0f044fd70b090f6862fb))
* Add thousand separator to conf matrix ([`4f98c0b`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/4f98c0b3be497748382d94a64ca698d67168bbd6))
* Add thousand separator to plotnine conf matrix ([`d27d0f7`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/d27d0f7dc390f6f88421070261399c8df5d505e2))
* Add lines to sens by time to event ([`b739267`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/b73926792969feb2c8ee4bc5d3ac1a857b1846bb))
* First stab at sens by time to event plot ([`91571f9`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/91571f9f5a1edba08ee33e170b98ede70a15d669))
* Add full performance figure ([`45c1d6e`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/45c1d6e73bb525c19f0864328c0a80b22fff6c5e))

### Fix
* Do not check for venv for tests, conflicts with CI ([`672d43f`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/672d43fa866fd6f29cec52abbb70da1cf194c052))
* Handle uneven number of plots in patchwork_grid ([`a833eb7`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/a833eb7d694e661c3c124d45f79c61095dfa4ef6))

## v0.73.0 (2023-05-17)
### Feature
* Convert auroc to plotnine ([`80f5cbf`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/80f5cbf734367a433687ca87fd026cd9f8ad73f2))

### Fix
* Incorrect path ([`925c94c`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/925c94c28435d7bb06fc017740c0fdef29a28898))

## v0.72.0 (2023-05-17)
### Feature
* Create plotnine confusion matrix ([`5045b67`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/5045b67b6b98e96fe55eedf5e5c12dd62cbf87b1))

### Fix
* Handle trailing commas in str_to_df ([`71331cf`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/71331cf53a6081a34c8e5da162e5dfde961549e1))
* Incorrect git import ([`629631d`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/629631d2e648b2f20fa0889b91dbab6bb153b35e))

### Documentation
* Improve docs ([`e6a1230`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/e6a123048465d74ee15570a0ec9681248026b816))

## v0.71.0 (2023-05-17)
### Feature
* Print a4 conversion factor ([`f6cde36`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/f6cde36b9f95153b12968175e5ec8531786b8409))
* Add patchwork grid functionality ([`4041220`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/4041220b5137e4d5a2a946a34c3333ac05549e07))

### Fix
* Autofix when creating pr ([`76470cd`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/76470cde6ce8305c66c153bc196dace8750a9cfd))

## v0.70.0 (2023-05-16)
### Feature
* Add action ([`0490672`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/04906728422ed5929c7e903f1b920d34afd1516f))

## v0.69.0 (2023-05-16)
### Feature
* First robustness plot ([`659d30d`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/659d30da095e1cdb6ead2d023a965d4743b8fec8))

## v0.68.0 (2023-05-16)
### Feature
* Split ci after bootstrap ([`f3e4f6f`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/f3e4f6f1a8a0e99cecb9be3ff98a3088c77ac01a))

## v0.67.1 (2023-05-15)
### Fix
* Run tox ([`183dc75`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/183dc75e453af6f6bd2444dbd7a9e605f766b015))
* Incorrect imports ([`26c08fd`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/26c08fdab2b248458c71036542ffcdbae2b182d0))

### Documentation
* Fix typo ([`5b8e3be`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/5b8e3be611f70ccac89840ee11a0fc97642ed491))

## v0.67.0 (2023-05-12)
### Feature
* Create pipeline and unified interface for evaluating the best run ([`d4fd7f3`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/d4fd7f3164b6439175147236dc371b5ce9beb2bf))

## v0.66.0 (2023-05-11)
### Feature
* Decrease bootstraps ([`2bd6500`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/2bd65009c623222ef327afb6d3756c371a9836c9))
* Add ci to timedelta plots ([`e6b9934`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/e6b9934b737d189df2fe88138a4ee67ee150ce40))

### Documentation
* Better explain utility func ([`46396a1`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/46396a1b3ad547d2f89ad59e941166071bcfee90))

## v0.65.0 (2023-05-11)
### Feature
* Add ci to timedelta plots ([`ce8c63f`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/ce8c63f52a120576aad61fe0c7274033c3bc5639))

## v0.64.0 (2023-05-11)
### Feature
* Handle only one true class ([`5a90247`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/5a90247d3c6056ca6525b1697d2c2340039934d1))

## v0.63.0 (2023-05-09)
### Feature
* Increase x-axis text size for base plots ([`b5ddf0b`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/b5ddf0bd48a868068a773bd4045a6a5bc73cbba7))

## v0.62.1 (2023-05-05)
### Fix
* Missing polars requirement ([`8e277e1`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/8e277e168bcdf02d1a20038f4defb6c26ee445d3))

## v0.62.0 (2023-05-05)
### Feature
* Allow str_to_pl_df ([`4cd53ac`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/4cd53ac0edb4954ec584c9d650c6fd8098389632))

## v0.61.0 (2023-05-03)
### Feature
* Allow custom splits for training ([`6e0bf71`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/6e0bf715c29806d87a1d4664470aa4f6815a2451))

## v0.60.1 (2023-05-03)
### Fix
* Get correct performance by ppr ([`09fa471`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/09fa471e2f4682be39985523e898453e6d05887f))
* Get correct performance by ppr ([`df468ea`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/df468ea98b377e37b67a30dac5c2d2fd1e7f9ca5))

## v0.60.0 (2023-05-03)
### Feature
* Preprocess text ([`b01cf35`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/b01cf355fab86cf29b9161fbce68c0509931289d))
* Performance_by_ppr ([`c7916fd`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/c7916fd3500a83008a442071f9c13a83973d16f8))
* Make n_bootstraps configurable ([`2af345e`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/2af345e0a98f5892714064d11567f2dd812d7a31))

### Fix
* Do not support multiclass in calc_performance ([`781692b`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/781692b8fec29ea8f650e50fd70afc7d6fb1990f))
* Assign sql cache if on local ([`2365b65`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/2365b6525a534143e29d5a0379af09158fc0cac8))
* Assign sql cache if on local ([`d57c9fd`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/d57c9fda77f2b3ba5a37f98f1b4b924109cc84c6))

### Documentation
* Add readme for evaluations ([`285f65d`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/285f65dea147c092599604ae989a44c0745f53e3))
* Add ([`bd0faff`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/bd0fafffb72c53a48a6f2f75fe5a3039cf2b9c0f))
* Update readme ([`7f3bbce`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/7f3bbce59af603cf32aa1e06375b5437374fa438))
* Update readme ([`2aa9b40`](https://github.com/Aarhus-Psychiatry-Research/psycop-common/commit/2aa9b4007388ba8012cababf4beab4b0d11d3ab6))

## v0.23.0 (2023-04-26)
### Feature
* Add logging and choose sfi types ([`d5f8e23`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/d5f8e23cfc2bdbc3ae9ff47ab89d409be454ea38))
* Create example scripts ([`76e063a`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/76e063a44ed88e18a23618ae45c887c26160c9fa))
* Initial text model pipelines ([`1934db0`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/1934db0d4fa615d7790920d1086fa18c6fa952b5))
* Add tests ([`d7a8bab`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/d7a8bab22e978b8db06e4bc13e561f989535c602))
* Initial simple preprocessing pipeline for all sfis ([`f941a4d`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/f941a4d3d69ad37b2901734fcd0b6d686621e9ce))
* Add include_sfi_name in load_text_split ([`4605c88`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/4605c88480254746ddf93e75c1ed13e6afc1d618))
* Include_sfi_name arg ([`58baf9a`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/58baf9aa7d732aee33ad47275fdbe57fc32ad7c8))
* Fit and load tfidf, bow, and lda models ([`3d33d9b`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/3d33d9b2a74f02afb375af71e287f2e01b8af19d))

### Fix
* Preprocess to one regex ([`c716653`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/c7166537bf596e05fd6c80b9ef32307c1f0a7f11))
* Remove symbols again ([`1210b7e`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/1210b7e4458ba0e83c4627fce2ef875d893b5bf6))
* Based on HLasses comments ([`32da48f`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/32da48f2b4f4a8ce8798489bfd9dff438e50ae8d))
* Insert model type in filename ([`1457387`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/1457387ded62189d256160f4d56f24f59ed96c11))
* Add doc strings to preprocessing functions ([`4e27650`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/4e27650fba042f154cfe219e9c1ded9a00cf34d7))
* Remove log.info and small fixes ([`84f3cc3`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/84f3cc34a48613fab60e98b1c7dc8ab70a09ba63))
* Ruff fixes ([`ea9c564`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/ea9c5643ef1e10e868e982d65f58081f60f94cfc))
* Return vectorizer and matrix + clean-up ([`e1c48a0`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/e1c48a089c42066444967f8808cfdbc4b26a1391))
* Query string ([`cb7424c`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/cb7424c576f9a0cbe7aa89c281fb3113e7c830e5))
* Naming and doc string update ([`141e52a`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/141e52a5c8b71c9e766c027cbe0c3e88380c86b5))
* General clean-up and change corpus in fit functions to list ([`22b6a9e`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/22b6a9e317436859ff2fad9f149345a9b19d673e))
* Change ngram default and clean-up ([`387f845`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/387f8452bc6d2646f24722e9582c9c323619f7f5))
* Small fixes to logging ([`c3a3f53`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/c3a3f53bd1c35a37c84cf3dcdd08a2d19cd0df7d))
* Remove old comments ([`4b88514`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/4b8851479a5d4b865cf7fc314a94320374aaea71))
* Change view name ([`a9bb0fc`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/a9bb0fc919454a3927e127c9d3bb7cddaf579699))
* Move save_text_model_to_dir to utils ([`469df3b`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/469df3b71299d6ec6e72dd08e406053ba242872d))
* Move save_text_model_to_dir to utils ([`26a80d2`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/26a80d2e961818eafa89611bb127a34083aba08a))
* Renaming in preprocessing ([`c381768`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/c38176857e0dd294b618763b14d69d4e3d05d35d))
* Remove stop_words arg and return models ([`3d29012`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/3d290123ad6d409e5444a90391786245b9d74cff))
* Change arg path to path_str ([`f781a74`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/f781a744c44aad1636ba6603ee50c82a2c551957))
* Enable multiple splits when loading data + add n_rows arg ([`8ae2d2e`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/8ae2d2ea166763973551d9b7733131b974c892c9))
* Remove Path from arg ([`29b442b`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/29b442b39be78071614b584f420d6cdde0fa80d7))

## v0.22.0 (2023-04-24)
### Feature
* Add feature descriptions for text features ([`84c696a`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/84c696a138ab0f589b2698f67c3097cb32d68200))

### Documentation
* Add readme link ([`217e550`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/217e5506d4e0b947355158c19854153fd2a48546))

## v0.21.4 (2023-04-04)
### Fix
* Remove unreasonably high or low bmi values ([`07f52c2`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/07f52c207c2c13fc79dd0ba97e243cef470f9fa0))

## v0.21.3 (2023-04-03)
### Fix
* Make sql query executable ([`e006490`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/e006490b5eb7a1dc4e2f0e902c17d5dcc5d19db9))
* Str turned into list of characters instead of list of words ([`0fae478`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/0fae47833085eb6c314ab02e86fde66ddfb82c27))

## v0.21.2 (2023-03-27)
### Fix
* Add unpack args to skema 2 wo nutrition ([`95c35c8`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/95c35c81991ae7df4b387f3edccb21c03d1aa8ca))

## v0.21.1 (2023-03-22)
### Fix
* Only keep weights above 0.5 kg ([`8a5a104`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/8a5a10401442337efd5bd33ce8bcf899fef4c642))
* Do not load invalid weights ([`7be4653`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/7be465301d57831a425f28f5eaedda1b783df9b9))

## v0.21.0 (2023-03-22)
### Feature
* Support new pipe annotation ([`a1bde17`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/a1bde17e354741a65bd85b8364b2cd45eeaab4d0))

### Fix
* Correct types ([`5cb0d5d`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/5cb0d5dede2746799a1be4c2950685b16a181b11))

## v0.20.3 (2023-03-14)
### Fix
* Set unpack_to_intervals to default ([`64391ca`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/64391cabcfd9e95d915f75f12fa64c4b81ee3365))
* Remove unintended space ([`9c6cd33`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/9c6cd33abd7f0b33c2497eebd2c2be2543890beb))

## v0.20.2 (2023-03-14)
### Fix
* Add skema_2_without_nutrition again ([`685c5cb`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/685c5cbc6188a5d516f3028adc37e5d1157064df))

## v0.20.1 (2023-03-11)
### Fix
* Cruft github action ([`c8f6278`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/c8f627877da2ac5e0b468b65e7bd9f9bfaacb483))
* Bug in cruft action ([`ec8267a`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/ec8267acc5888a3f402fb6f032a7d301cddb12b9))
* Remove psycop-ml-utils, no longer exists ([`d8fbb65`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/d8fbb657d1bbe439a5863edfdbe7caa9a0fd2010))

## v0.20.0 (2023-03-09)
### Feature
* Add more glc loaders ([`b765e77`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/b765e77ff32e6d4142ab63b6cc8e47a870946322))
* Add type 1 diabetes loaders ([`b682984`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/b6829848180a8d2b91dfd046049c8e216fca3689))
* Make sql loader verbose ([`602f4f3`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/602f4f388c4015330f5c789510e399fc4a2cd878))
* Add caching to sql_load ([`a68c15d`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/a68c15d3a0a994987ccb120bbb2bd251c28c00fc))
* Ibid ([`46da732`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/46da732d602e7421e0677f12bf835c0d96fb3c31))
* Add support for keeping code col when loading diagnoses ([`51ca63e`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/51ca63e72a420ecd49065e9cccb319bbfb2012ab))
* Add t2d diagnosis loading ([`6b8231c`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/6b8231ca4c96b24f74013d85f815caab7ae2123f))
* Add ogtt ([`f6c07a9`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/f6c07a9bc15f7475fda9ad325d082b43458f25fb))
* Update current blood sugar measurements ([`5e8051a`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/5e8051a7db28ba8e2957b1d01a6f5cb91eba988b))

### Fix
* Lacking prefix on loading glc ([`d9bdbcb`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/d9bdbcb752732b92c77d61ede187a7b41aba6520))
* Inappropriate matching ([`e2409ed`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/e2409edb280c4307f24fcbae82e46eadcb5e3e96))
* Poetry formatted dependencies ([`125500a`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/125500a72ec832ed1773115f88ba251c99fcf553))

## v0.19.2 (2023-03-06)
### Fix
* Disable cache ([`0242114`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/02421147d494eb693f41538778c76d040060930b))

## v0.19.1 (2023-03-06)
### Fix
* Drop rows with NaT ([`5a1d908`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/5a1d9080a05a28748ef45df7c61e5aa9af5de77a))
* Round timestamps to whole seconds befor droppig duplicates ([`e503bf3`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/e503bf3b67f2797dfc31b669502fcb2a7468e35d))

## v0.19.0 (2023-03-03)
### Feature
* Add option for which timestamp to get when loading physical visits ([`ef369b8`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/ef369b872b41230108877e41cbb490e6cfe1a5e4))

### Fix
* Drop duplicates in the output_df ([`636cc48`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/636cc488762eb31f97a4ba0e334a245629b7fbe7))
* Don't load duplicate visits ([`5028b1d`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/5028b1da604cd8f66c7bb4e8ed69642d622c57b0))
* Physical visits should only load physical visits ([`b7c50cf`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/b7c50cf972da0ea40d9c75eecee879a281ead7cf))
* Did not rename to timestamp before returning ([`f43522c`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/f43522c9a83d014fafae1d13b0e59ec037b940d3))

## v0.18.4 (2023-02-22)
### Fix
* Loader names still too long ([`3321b88`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/3321b88961779d4cee3774c566ca7793d023b7fc))

## v0.18.3 (2023-02-22)
### Fix
* Loader names too long for wandb ([`cc14da2`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/cc14da233e7764f148b9966047628fea0bad4a1e))

## v0.18.2 (2023-02-21)
### Fix
* ValueError correction ([`595479e`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/595479ebce800810327674ae79e58a2eb78b877e))

## v0.18.1 (2023-02-15)
### Fix
* Adjust function for saving integrity checks ([`de2577e`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/de2577e3add215c4ddf65a8c047c04ed9bb962a4))
* Restructure overarching description func ([`54c24a2`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/54c24a2429e03455138573d56a83af0238454778))

### Documentation
* Better function description ([`7eb9e54`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/7eb9e5433a21fc8a5cb78e63e6aa9501edaec785))

## v0.18.0 (2023-02-14)
### Feature
* Add arg for choosing timestamp and add warning ([`159a176`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/159a1764aed4c6f3a13d2f6f92d2dad83a957338))

## v0.17.2 (2023-02-13)
### Fix
* Make naming scheme consistent ([`c125b48`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/c125b4889f7e3f2e39d13e47fedd4c9ed0bf4e48))
* Attempted rename of unspecified df ([`c266bd8`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/c266bd8611ac839264efcba4dc17362879871cda))
* Revert logic ([`ad110ee`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/ad110ee72dd9994dd2fae1cedc8a4ff63dde1c93))
* Quarantine_df and quarantine_days can be left as None ([`f130370`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/f130370eaa33ecffd10487392206a6acaf96d09f))

## v0.17.1 (2023-02-10)
### Fix
* Allowed types works again ([`dbe75ca`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/dbe75ca79c9c596be06fbd7649d7f110ac351bc9))
* All arg names now congruent, visit_types takes a list of visit types instead of string ([`e63e9d4`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/e63e9d43ff1b1f0793e4f3940ee8d45a22d76c90))

## v0.17.0 (2023-02-09)
### Feature
* Add text loaders ([`9c7d959`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/9c7d95968c3181d7dcbfc3506961a77f2dd7768d))

## v0.16.1 (2023-01-31)
### Fix
* Use acute outpatient visits as well ([`659af23`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/659af23622848290e83967e8fe528bc4f999ceff))
* Typo, and use newest data ([`bbbc8f5`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/bbbc8f53c69106e0a1c188d163786cb181ef4483))
* Use end dates for all contacts ([`d8940c1`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/d8940c1717a0f97dee8ff593bf7b9cee6d99b402))
* Use end times for all diagnosis loading ([`4d9e600`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/4d9e600e70744b7ebe46458bc2c1c08b6102f86a))

## v0.16.0 (2023-01-27)
### Feature
* Remove try/except to avoid debugger getting stuck on it ([`3884ab8`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/3884ab89cc15626385689530ce9ea15c56508aff))

### Fix
* Move all str operations into the if statement ([`91f9174`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/91f91748cca722e1d07113ffbadbaed0d0a8a925))

## v0.15.0 (2022-12-19)
### Feature
* Move logs next to their dataset ([`e0ed033`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/e0ed033a71f93f3a9b642db79b03c2017a067b33))

### Documentation
* Improve quarantine docs ([`1b23f19`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/1b23f19055743d8d0f54b413bf9c5a84a31322de))

## v0.14.0 (2022-12-16)
### Feature
* Name wandb project_name-feature-generation ([`b601d80`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/b601d80b1a3e2936ddff31316dd85d6d08c79584))

## v0.13.0 (2022-12-16)
### Feature
* Improve logging in flatten_dataset ([`63f252f`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/63f252f3fe825a6f9bd6989e5ae515ae848eb8d7))
* Enable minimum specificaitons ([`669e3ed`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/669e3ed34631ad9c5c481b42f2263ee12f8a070d))
* Enable minimum specificaitons ([`523cfd1`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/523cfd114c422bd2edf4f4cdd35b1b808a2f5b92))
* Log rows dropped by PredictionTimeFilterer ([`7e02d8e`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/7e02d8e6065989c944045d4cdba571d425beb1db))
* Add moves loader ([`0521dd0`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/0521dd00ddc2ff614311102858eea0bf9ea86696))
* First stab at loader ([`f9048b8`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/f9048b8741b63c3781d18be46f9b42c8550bd5e0))

### Fix
* Add pred_time_uuid if not specified when filtering ([`acca5b9`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/acca5b9b557200517fa148ec97f9613f2d765a11))

### Performance
* Avoid groupby in filter_prediction_times ([`a66e361`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/a66e361fdcf3ce75a9d25a3d5b732c72deb2abb0))

## v0.12.0 (2022-12-15)
### Feature
* Add rows dropped logging ([`33ba525`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/33ba525c1cd3231dce080b64eb57d571a824ba3b))
* Allow filtering based on quarantine dates ([`3deb052`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/3deb052d7373ba8a4ebef265f8364101555bfd72))
* Improve logging - debug to file, info to stdout ([`aff10a9`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/aff10a9b4f146509252f905bcc38ea79d6575992))
* Move wandb init earlier so wandb_alerts can cover values_df loading ([`6c153b1`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/6c153b15815e063af85a8f1c6bf0696bb67519e0))
* Generate full feature set ([`9ba907a`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/9ba907aa3134a32127cd115f2efa3c53041b713a))
* Wrap as much of main as possible in wandb exception ([`3b085af`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/3b085afee6e92409e10aca323821bc481e076d9a))
* Allow timestamps only return from visit loaders for use as pred_times ([`f9534e0`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/f9534e00df5249c15f2c0eaca10bd6ed72681a6e))
* Migrate some loaders to logging. ([`f81fd92`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/f81fd921d17faeb28e926807909c9036ee418317))
* More explicit logging ([`7969210`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/7969210c9c85f07fbda485e8076e27f90901d072))
* Init changes ([`f257daa`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/f257daa832a26b5508a67a7eadb9040075bc801a))

### Fix
* Use lookbehind instead of interval days ([`7e14ad5`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/7e14ad59583065d7be8c344cc6d7cab26066e83a))
* Only one feature cache per project ([`cb0b8b0`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/cb0b8b0709fbca2ea1bcec5054ea8aac9ad39faf))
* Unused input args ([`fa14461`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/fa144615ecf71a88d3c6e0be93b301b67977abe2))
* Wandb util was missing text kwarg ([`64c1729`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/64c1729f2ec696fe913a2851880168513797fa5a))

### Performance
* Infer CPU cores from logical cores ([`309e9d2`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/309e9d26d4d902fb0492ba99bdda38776aab24ab))

## v0.11.0 (2022-12-13)
### Feature
* Add wandb alert on exception ([`3ff6e37`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/3ff6e3797910bb159e23496da6d863097b646267))

### Documentation
* Improve create_flattened_dataset docs ([`637edfe`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/637edfe86b49540e44f7b8879e57e9be82890b10))
* Misc. docs ([`4eac2ba`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/4eac2ba37913852e86e47cb679379f5678af8783))
* Fix github test badge ([`dffeedc`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/dffeedc33c3ff7360eac67d4edcb0ff2c05fb3e0))

## v0.10.0 (2022-11-21)
### Feature
* Add n_hba1c_within_n_lookahead_days ([`e84b591`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/e84b5918724a55b721ec4d1a7291533227fe9ef8))
* Add outcome ([`cd39dd6`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/cd39dd6adfaa0c261abb2942ac9f215670c1c92d))
* Add birth year as a predictor ([`7b186d2`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/7b186d2fc339dd423207b9311cdb6d1fad7078ee))
* Allow exclusion of specific atc codes ([`75619a1`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/75619a122e26ad43fd7058e3db49c062e33b0b9f))

### Fix
* Date of birth col name should respect output prefix ([`6ec6535`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/6ec6535a2df4161ffc6e94e02eb9b340722f43e7))
* Incorrect column name when adding age as predictor ([`cdbf25c`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/cdbf25cd26f60baa795e43bc9df3865868248960))
* Errors in sql loaders after refactor ([`28c9f63`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/28c9f63fd8b81892fbea2695df94df47f6fe8dc6))
* Correct type hinting in load_diagnoses ([`f2d5c5b`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/f2d5c5bfebce3fc8c3c61ee5231716dfc7883c8e))

### Documentation
* Speccify that n_rows = None returns all rows. ([`a4720a8`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/a4720a8777601e81993f6707a4f4f48a6f850282))

### Performance
* Shuffle feature specs to even out compute vs. IO load ([`0db9f0f`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/0db9f0fd77989fdced4489ca9c45caff3d741086))
* Tweak n_workers for more performance ([`3eeee4d`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/3eeee4da7092364d68a8a6eb2e3e028df4403fa1))
* Segment feature loading for more parallelisation ([`9ee5c87`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/9ee5c8778820da29d370653ce435665226e3cfdb))
* Rotate feature addition for debugging ([`76af9c7`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/76af9c717059f063d8aeb6756816b8e574bb845b))
* Parallelise temporal predictor loading ([`8d53f16`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/8d53f165e760e581d8888287474f6f353642ae0b))
* Only create one subprocess per values loader ([`1a3e5de`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/1a3e5dedb66a864b27be5318359b60f778eaa15b))
* Parralelise groupspec combination creation ([`9ccba2a`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/9ccba2a24538b752409f166f82a0474805e18150))

## v0.9.0 (2022-11-18)
### Feature
* At groupspec init, iterate over values_loader and check that they exist in the loader registry ([`04dfd7e`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/04dfd7e7e038472cfd26f67c79a6b050cc13b15e))

### Fix
* More explanation in error message ([`b784991`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/b7849911c85ca6ac5bd165b7a48ccce1a768f70b))
* Bettee valueerror message formatting ([`7b3b994`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/7b3b994cbe38df73a4149c4463b5f283ad297218))
* Better valueerror message ([`d92f798`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/d92f7989af27a879fd090bed33ce5027e96e581b))
* Find invalid loaders ([`ba2d4c5`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/ba2d4c540f097c33ca5c29a0b72a908ad6dc04e3))

## v0.8.0 (2022-11-17)
### Feature
* Allow load_medications to concat a list of medications ([`d78f465`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/d78f46592213b8245229d6618d40f1a1ff4d80eb))

### Fix
* Remove original functions ([`da59110`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/da59110978469b0743ce2d625005fc90950fb436))

### Documentation
* Improve docs ([`9aad0af`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/9aad0af6205af2e3deffb573676af5a20401bae1))

## v0.7.0 (2022-11-16)
### Feature
* Full run ([`142212f`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/142212fc63a59662048b6569dc874def92dfe62f))
* Rename resolve_multiple registry keys to their previous one ([`3fd3f35`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/3fd3f3566a8a9312ef9a8326a700b162ed9815c3))
* Reimplement ([`c99585f`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/c99585fdf9f9f407a69e0ead05f935d34ed86a63))
* Use lru cache decorator for values_df loading ([`4006818`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/40068187da20854fcca980872bc42b8a3a096cc9))
* Add support for loader kwargs ([`127f821`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/127f8215c35b792390595b890210baa0e8cf3591))
* Move values_df resolution to AnySpec object ([`714e83f`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/714e83fd3722b298cdd256b06915659ca7a34259))
* Make date of birth output prefix a param ([`0ed1198`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/0ed11982ba1b239e5650d23dbfab707100e38137))
* Ensure that dfs are sorted and of same length before concat ([`84a4d65`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/84a4d65b731a6822d0a8f6313d01b7de9c574afe))
* Use pandas with set_index for concat ([`b93290a`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/b93290ae733857855abe8197291dd047cf6c6fa8))
* Use pandas with set_index for concat ([`995da41`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/995da419baef8fdb1f205610d63805c152156474))
* Speed up dask join by using index ([`3402281`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/34022814b6e9c93a715a2d6343f7c038feb6a932))
* Require feature name for all features, ensures proper specification ([`6af454a`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/6af454a325bdb07a37c435246b0ead4d4dad971b))
* First stab at adapting generate_main ([`7243130`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/724313073d5eb225b3eddba597064f35053b0bd4))
* Add exclusion timestamp ([`b02de1a`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/b02de1a92f12545bc1ac0ea40f98468f21185259))
* Improve dd.concat ([`429da34`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/429da346b0de1e07809176a1d2d34962c7e9770a))
* Handle strs for generate_feature_spec ([`7d54488`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/7d5448853ba3bdd0b13071afbb2c738d741337d3))
* Convert to dd before concat ([`06101d8`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/06101d86561af56eebaea2090baaf27aa3747b71))
* Add n hba1c ([`3780d84`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/3780d841699d2a6b9077ca4fa3117d69f32bb123))
* Add n hba1c ([`614245e`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/614245ead3fcc5b554a26ba515ff689d2627429b))

### Fix
* Coerce by default ([`60adb99`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/60adb999c83b6d93d97f1c6537f20c012721561e))
* Output_col_name_override applied at loading, not flattening ([`95a96ce`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/95a96ce64a186c01f4e4e09d8787a97e42388df8))
* Typo ([`01240ed`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/01240ed7b06843011593bcb3c3c71283918c90b2))
* Incorrect attribute addressing ([`a6e82b5`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/a6e82b59ca353413066346e089f1557dc831d145))
* Correctly resolve values_df ([`def67cd`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/def67cd954440df76f1570acf7e48f68ae636d6c))
* MinGroupSpec should take a sequence of name to permute over ([`f0c8140`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/f0c814017b6f355d5916ba15fe26d9f3350a3a7b))
* Typo ([`61c7241`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/61c7241d11f7bff3bad11e98cfea38600e239167))
* Remove resolve_multiple_fn_name ([`617d386`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/617d386095139bc3445a5f4d14ffebce1e5ffa24))
* Old concat resulted in wrong ordering of rrows. ([`3759f71`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/3759f719070175c8be4184a0bdc5fc07db2c492c))
* Set hba1c as eval ([`89fe6d2`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/89fe6d209b93d345d9a0d8cd562e90ec395dfa8d))
* Typos ([`6eac440`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/6eac4408d8f0a58bb4cc66ac948bae5519a2c8cd))
* Correct col name inference for static predictors ([`dfe5dc7`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/dfe5dc72d5d22332ce3d496fb1d3bcca3c9328c7))
* Misc. fixes ([`45f8348`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/45f83488bef809ae059825caea9bf6937a5264d9))
* Generate the correct amount of combinations when creating specs ([`c472b3c`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/c472b3c69e0dfc64b433546e538298ddd2d44a5f))
* Typo resulted in cache breaking ([`fdd47d7`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/fdd47d705f166fcc3dc54612dc0387761d0489a9))
* Correct col naming ([`bc74ae3`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/bc74ae3089a7bbfc99ee31d82902e1c98e30f18e))
* Do not infer feature name from values_df ([`150569f`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/150569fde483f6c427f1efe5688038340dfceb92))
* Misc. errors found from tests ([`3a1b5db`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/3a1b5db493566592b349d317f7641d7564a662ad))
* Revert falttened dataset to use specs ([`e4fada7`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/e4fada7a9fb98d1ebccd6c41568619aa7e059d79))
* Misc. errors after introducing feature specs ([`0308eca`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/0308ecae8032ff309725b0917fd3901fadf102f9))
* Correctly merge dataframes ([`a907885`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/a907885f592ba345cdf68ce5299699aacdc97b49))
* Cache error because of loss off UUID ([`89d7f6f`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/89d7f6f0ce557c7c3126116864ba75d0ddb0037e))
* New bugs in resolve_multiple ([`5714a39`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/5714a39c9e84081f6429dd0b8119873a9610e804))
* Rename outcomespec appropriately ([`41fa220`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/41fa22069453ac6df7dae824d49944775cf12ecc))
* Lookbehind_days must be iterable ([`cc879e9`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/cc879e9d6b0f806a2a604ff71cb3febbd625c2aa))

### Documentation
* Document feature spec objects ([`c7f1074`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/c7f10749d49b14a4614436097de2478f3e7fc879))
* Typo ([`6bc7140`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/6bc71405a318de4811f259b2823c91f1951ebb95))

### Performance
* Move pd->dd into subprocesses ([`dc5f38d`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/dc5f38db7d09900955e475d9c87837dab207ba9b))

## v0.6.3 (2022-10-18)
### Fix
* Remove shak_code + operator check ([`f97aee8`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/f97aee8ff932270abed737308591cc87678062a8))

## v0.6.2 (2022-10-17)
### Fix
* Ignore cat_features ([`2052505`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/20525056d6e97aceb277a5e05cde3d8e701650e3))
* Failing test ([`f8190b4`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/f8190b47b020782e1029f875bc3acee5c3abe566))
* Incorrect 'latest' and handling of NaN in cache ([`dc33f7e`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/dc33f7ef68c065814779f44b7dd8e65c46755fea))

## v0.6.1 (2022-10-13)
### Fix
* Check for value column prediction_times_df ([`5356464`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/5356464ee5dbe302cf2bafd3203be88016e6bcaf))
* Change variable name ([`990a848`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/990a848a7d63410d06e491664d549f04a24a4384))
* More flex loaders ([`bcad700`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/bcad70092069cb818a67383bd8a925248edf04cd))

## v0.6.0 (2022-10-13)
### Feature
* Use wandb to monitor script errors ([`67ae9b9`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/67ae9b9ebecef68d4d0ceb74b58dc7bd3f6798b6))

### Fix
* Duplicate loading when pre_loading dfs ([`7f864dc`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/7f864dca9315b296e16cc1c9efd84e73627c9e2f))

## v0.5.2 (2022-10-12)
### Fix
* Change_per_day function ([`bf4f18c`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/bf4f18c10c66b8daa660d9ad9bb0dd05361dde75))
* Change_per_day function ([`b11bcaa`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/b11bcaaaac0e8de75e798491b0e4355220029773))

## v0.5.1 (2022-10-10)
### Fix
* Change_per_day functions ([`d696389`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/d6963894c458cdacc43cec579af1452a427ab86f))
* Change_per_day function ([`4c8c118`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/4c8c118e9f0e53c145ad07132afcc475890cb021))

## v0.5.0 (2022-10-10)
### Feature
* Add variance to resolve multiple functions ([`8c471df`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/8c471df351855a5f7b16734f999c73ae0e590874))

### Fix
* Add vairance resolve multiple ([`7a64c5b`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/7a64c5ba6d776cea6bf7b8064698bf9ad4d6814e))

## v0.4.4 (2022-10-10)
### Fix
* Deleted_irritating_blank_space ([`a4cdfc5`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/a4cdfc58ccf7524a308af1bab3b0ca6f0b15e834))

## v0.4.3 (2022-10-10)
### Fix
* Auto inferred cat features ([`ea0d946`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/ea0d946cbf658d8d7e22d45363f9dd7d5a7e3fff))
* Auto inferred cat features error ([`f244715`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/f2447156beef5128819f97f7a9554d03d394e01a))
* Resolves errors caused from auto cat features ([`667a905`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/667a9053f89413ada54624ae19d0d7e880724573))

## v0.4.2 (2022-10-06)
### Fix
* Incorrect function argument ([`33e0a3e`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/33e0a3e959a2cf864c2494810741b02d073c55c4))
* Expanded test to include outcome, now passes locally ([`640e7ec`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/640e7ec9b0ed294db2e58ae56d1a06740b4e8855))
* Passing local tests ([`6ed4b2e`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/6ed4b2e03f42f257342ae62b11302d76449a1cdc))
* First stab at bug fix ([`339d793`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/339d7935c0870bbdd140547d9d3e63881f07a6e8))

## v0.4.1 (2022-10-06)
### Fix
* Add parents to wandb dir init ([`5eefe3a`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/5eefe3aa14dbe2cd3e8d422c0224f3eb557da0df))

## v0.4.0 (2022-10-06)
### Feature
* Add BMI loader ([`b6681ea`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/b6681ea3dc9f0b366666fb4adb964d453c094844))

### Fix
* Refactor feature spec generation ([`17e9f16`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/17e9f166aa48b2ed86f4490ac97a606232e8aeaa))
* Align arguments with colnames in SQL ([`09ae5f7`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/09ae5f7b91523c53431e6ef52f3ec6b382b70224))
* Refactor feature specification ([`373b0f0`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/373b0f025d4d74bc0041c3caa2ef8cf7559888ff))

## v0.3.2 (2022-10-05)
### Fix
* Hardcoded file suffix ([`0101acc`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/0101accb995d060908b28f1338a313d82661683a))

## v0.3.1 (2022-10-05)
### Fix
* Mismatched version in .tomll ([`292979b`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/292979bf85401818d5837a159c30c88c67ac454d))

## v0.3.0 (2022-10-05)
### Feature
* Update PR template ([`dfbf153`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/dfbf153348594b8b0eaac0974fff7c69680c473d))
* Migrate to parquet ([`a027549`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/a027549cd1bc17527c8c28726748b724b639d510))
* Set ranges for dependencies ([`e98b2a7`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/e98b2a708356b167102fcf3f77bf1f623f34bf07))

### Fix
* Pass value_col only when necessary ([`dc1019f`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/dc1019f6f42510ea9482c1ad83790908b839ed15))
* Pass value_col ([`4674e4a`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/4674e4aef272469a1b68baab6656fba7d5b6b046))
* Don't remove NaNs, might be informative. ([`1ad5d81`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/1ad5d810cc7ea969ce190e13b7b4cb25be15de01))
* Remove parquet default argument except in top level functions ([`ec3a98b`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/ec3a98bca22bf8385a527cefd2c80dd80b3a60ff))
* Align .toml and release version ([`80adbde`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/80adbdeec8cde7b8c0b5e37393f2b48844c53639))
* Failing tests ([`b5e4321`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/b5e43215943777ffa5ac9d63f878b0a2358485cd))
* Incorrect feature sets path, linting ([`605ccb7`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/605ccb7c5a3cfb103efcda8f965e8a72ae52ae7f))
* Handle dicts for duplicate checking ([`34524c0`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/34524c055f1335ae703fbfce11f234c065c4ccb9))
* Check for duplicates in feature combinations ([`63ad162`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/63ad1628f750abdd58c24d9b6ea53a9be8ef6032))
* Remove duplicate alat key which prevented file saving ([`f0c3e00`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/f0c3e006c84cd41054fdbca4cf1266d9f393a059))
* Incorrect argumetn ([`b97d54b`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/b97d54b097986f452ae2f00f5bba2a6f051c1132))
* Linting ([`7406288`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/7406288d50ecfe9436f95726a6fd72c886478923))
* Use suffix instead of string parsing ([`cfa96f0`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/cfa96f0d768c1fbbeca372f93ab970535479f003))
* Refactor dataset loading into a separate function ([`bca8cbf`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/bca8cbfb861aecc995e657285a0ad4011b47e407))
* More migration to parquet ([`f1bc2b7`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/f1bc2b7f872ed17c28501acdb377cf385bbe9118))
* Mark hf embedding test as slow, only run if passing --runslow to pytest ([`0e03395`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/0e03395958f30d0aff400d7eb1f227808f57226c))

## v0.2.4 (2022-10-04)
### Fix
* Wandb not logging on overtaci. ([`3baab57`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/3baab57c7ac760a0056aefb95918501d4f03c17a))

## v0.2.3 (2022-10-04)
### Fix
* Use dask for concatenation, increases perf ([`4235f5c`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/4235f5c08958ac68f5d589e3c517017185461afa))

## v0.2.2 (2022-10-03)
### Fix
* Use pypi release of psycopmlutils ([`5283b05`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/5283b058bc67ac4a4142aaaa9a95a06f5418ef01))

## v0.2.1 (2022-10-03)
### Fix
* First release to pypi ([`c29aa3c`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/c29aa3c847bcdafbc8e60ff61b6c2218ab8c1356))

## v0.2.0 (2022-09-30)
### Feature
* Add test for chunking logic ([`199ee6b`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/199ee6ba62cd915b3885ad5101286d6caca7a72f))

### Fix
* Pre-commit edits ([`94af649`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/94af64938a1ba082a545141ed5d332dbdd1df867))
* Remove unnecessary comment ([`3931395`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/393139512dd58ebeec143499317425ca63b25e45))

## v0.1.0 (2022-09-30)
### Feature
* First release! ([`95a557c`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/95a557c50107b34bd3862f6fea69db7a7d3b8a33))
* Add automatic release ([`a5023e5`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/a5023e571da1cbf29b11b7f82b7dbb3d93bff568))
* Update dependencies ([`34efeaf`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/34efeaf295b468c3ebd13b917e37b319df18ccf6))
* First rename ([`879bde9`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/879bde97033e627269f3ffe856035dfbe1e1ffb7))
* Init commit ([`cdcab07`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/cdcab074310c843a7e1b737d655136e95b1c62ed))

### Fix
* Force dtype for windows ([`2e6e8bf`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/2e6e8bf148db256f6a047354a474705c25af3156))
* Linting ([`5cdfcfa`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/5cdfcfa75a866919364bd5bbf264db4fcaa8fdda))
* Pre code-split import statements need to be updated ([`a9e0639`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/a9e06390aba1fa5cdcb7d0e9918bc158dbdcaf26))
* Misspecified python version in action ([`fdde2d2`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/fdde2d2e2bc7f115a313809789833bcd8c845d6d))
## v0.0.1 (2023-03-30)
### Fix
* Typo ([`9197a4e`](https://github.com/Aarhus-Psychiatry-Research/psycop-model-evaluation/commit/9197a4e6bf5d538ebaa8f7b9fad5d248f39a8c3a))
* Erroneous imports ([`a6569b5`](https://github.com/Aarhus-Psychiatry-Research/psycop-model-evaluation/commit/a6569b57d4e63bf0fd39e5dfa6c6814e1fe9d5d4))
* Erroneous imoprt ([`d246fcf`](https://github.com/Aarhus-Psychiatry-Research/psycop-model-evaluation/commit/d246fcfdfc58a2b6b351705d0ea02e6946575f49))
