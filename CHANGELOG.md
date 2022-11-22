# Changelog

<!--next-version-placeholder-->

## v0.21.0 (2022-11-22)
### Feature
* Allow overrides when loading cfg as pydantic ([`5c58a7e`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/5c58a7e50559df645e571e739b45e24a69b86e60))
* Add logging of superprocess ([`1cb5d92`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/1cb5d92865a2a8de7eb296ce404cff77da32b871))
* Add random delay to decrease resource competition ([`9d146ef`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/9d146ef5a48f16386995bd4f029765ba57e7557a))
* Dynamically infer outcome column names ([`fa3c168`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/fa3c16842c53e29cf7cdbaf4d0301fc3a97567cb))
* Drop datetime predictors ([`3203d14`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/3203d14fc41f6470b3e9e48cb627f559cd4d10d7))
* Also load unfiltered df in dataset inspection ([`ef1a954`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/ef1a95466a795bb5bf6c3a7245ec0605a0b88754))
* Use exclusion criteria ([`11a7058`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/11a7058ea75e4b5c95511d4605b7696b7480877a))
* Add example of dataset inspection ([`bb38f11`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/bb38f11b01cd9231522ee5ca7f8050b977a932e1))
* Update config schemas ([`ecb3964`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/ecb3964d91fd00a575f68a39f611415a6247b99d))

### Fix
* Cyclic time plots override one another ([`df8510d`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/df8510df30aa1ee984dab134b742ea6324674298))
* Convert negative values to nan ([`516c42b`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/516c42bf71d061dc519533a2257b78be79f581bb))

### Documentation
* Pylint ([`0a6a0e4`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/0a6a0e4b40f7227e9a7a79cc78f51d9d81dd14b4))
* Expand docs ([`0bfcd4e`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/0bfcd4eb8590355c02d57d1993da43e377633069))

## v0.20.0 (2022-11-22)
### Feature
* Performance by citizen ID ([`9a2031a`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/9a2031a60ae7e494bb2270e835d13bd4453e7e99))

### Fix
* Tests ([`7dfc2b1`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/7dfc2b12b58a113c2f10707801813aa7c85d49c7))
* Wrong argument passed ([`de58ac6`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/de58ac6396b31ee06ac11fbc5ed31dbcb1537da5))
* Test works ([`d8233dd`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/d8233dd22dab63e4ef571ee183f623c61ed7195b))
* Create test ([`729bf81`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/729bf81f661f96eee8b2c2e58e9d586ddd3b5745))

## v0.19.1 (2022-11-21)
### Fix
* Update integration testing readme instruction ([`c4d54c3`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/c4d54c32bfb39096e5ae82ee60b8bef9815043f2))
* Naming and logging of eval_dataset.parquet to disk ([`052c9f0`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/052c9f06d37b77b5c135ec447bd4926304b99a47))

## v0.19.0 (2022-11-14)
### Feature
* Add month of year ([`826961c`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/826961cda1217322714d3b6ec29db1df3f0fc9c0))
* Performance by cyclic time ([`348e2f8`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/348e2f863cb09d6962b0d9423b383b4542066c3f))

### Fix
* Review comments ([`447cfb9`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/447cfb947a5a55759e63df70ae5718a37120694c))
* Parameterise tests ([`bf37a95`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/bf37a95f82ef411bdd1920bf5b44d8a2882cbd53))
* Add plots to applicatrion ([`a2edd96`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/a2edd9658b2a03cac2dca114bdbaeb2eea4f8fdd))

## v0.18.0 (2022-11-14)
### Feature
* Feature sel for NAs and negative values ([`9391f41`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/9391f417d67cb3cafdd928b06999d1c29fdce914))

### Fix
* Add documentation to cfg file ([`0976ee1`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/0976ee1fbcdd66ecdf1ea91834e4cbd61024e8e1))

## v0.17.0 (2022-11-14)
### Feature
* Adjustments to eval ([`81c0f93`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/81c0f9314c839485b847dfbeaad50210c02f307e))

### Fix
* Smaller xticks fontsize ([`2a00325`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/2a003251bbb7df70a7d3cba8e95b30c89e272019))
* Review comments ([`f5e5a42`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/f5e5a4250ddbf894a66d26306727844e8e748c18))
* Remove unused arg ([`a60659f`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/a60659fc2b229ab79e494a8b9492523bce74e571))
* Remove unused args ([`51e6b74`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/51e6b74241f6c72d71c1b5ef8b51c79d0d3c94b6))
* Keep two decimals on heatmap annotations ([`cd498e3`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/cd498e33fc464d9ce2eb82d6e9a00da6290f0691))
* Remove automatic bin trimming, not needed ([`beae28b`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/beae28ba9e05f9aaafd27fe8be2679ea989801fb))

## v0.16.0 (2022-11-11)
### Feature
* Add auc roc plot to evaluat_model ([`b1fff1b`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/b1fff1bd1e79d4b8e3085270f819cb78f38c4615))
* ROC-AUC for eval ([`4bb748e`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/4bb748ea932e1c1e3a3abd105e5d49f1292a3bbe))

### Fix
* Remove plt.show ([`f50aa1d`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/f50aa1d1220812306e849bb71bae4a5193014ab4))
* Unused argument ([`e2adf61`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/e2adf617db649fbdc2180749fd75cbda1db8ba06))

## v0.15.0 (2022-11-01)
### Feature
* Add min_age ([`3ebd3fe`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/3ebd3fec7cc5c2da54b9de8ebb0dcf7be9fa54cb))
* Add exclusion timestamps to dataset processing ([`cbc34a6`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/cbc34a68eee6e0916a997f0b171e3b1943e8454d))

### Fix
* Old dw_ek_borger ([`0a7a23d`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/0a7a23d011edcfa7921c7da79b2a350e476786cf))

### Documentation
* Linting ([`d349339`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/d3493391341ec2d8e9bc64f97ff942b9d69dcc68))
* Improve config docs ([`4d77b2f`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/4d77b2f143fbfaebdc5c2ac9fd02f862333f0efd))

## v0.14.0 (2022-10-28)
### Feature
* Generalise performance by input while retaining evaldataset ([`9d0a583`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/9d0a5832ab83a8296c3911dfae6d63d85c3e43b6))
* Add the plots to default plotting ([`f48981b`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/f48981b61a20de814d133da70b19bc0bd51ec06a))
* Adding performance by age ([`350850a`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/350850a90b85f57293be5fc2b26e04dcd4a332f8))
* Add performance by N HbA1c plot ([`d50510e`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/d50510eab66bb4c05405777e194f00a906874092))

### Fix
* Yaml ordering ([`be21601`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/be21601231bc54b0b5b26d20023430a92ba3b1f0))
* Check if tuple ([`9c58559`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/9c585598127d37a285e71cf82a9669960e93881d))
* Check if tuple ([`2ad86e1`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/2ad86e1a4c2f71e6091a36f1780a71c4249fbf70))
* Test works ([`cb62855`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/cb62855ed8ea94dc086829db2cc0c445cc34aade))
* Pretty bins function ([`a840aeb`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/a840aebf63b13fbad8d93b527e0472fe34e46505))

## v0.13.0 (2022-10-28)
### Feature
* Automatically infer whther GPU is present ([`407521d`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/407521dc26aa5ff9ec783c5e50cb30bf94f12a7d))

### Fix
* Remove incorrect toml header ([`58c99f3`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/58c99f31ae496dd23761b9c7a5be64e46b73b5df))
* Type error and import error ([`125081d`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/125081d81a50df076bfabe463f9207d9e03a6a38))
* Dict fixes and circular imports ([`152a376`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/152a376ccbfd3b86e135f8abfbda2e7cbbdaca07))

## v0.12.0 (2022-10-26)
### Feature
* Add watcher to main training script ([`917d42e`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/917d42e08c0e51b5ae6e7c5271eb827409613fb4))
* Make watcher store separate max performance per lookbehind/lookahead combination ([`091a3c0`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/091a3c02af4150dcccb46c60753f0b7380e5f588))
* Add watcher ([`bb63f53`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/bb63f531cd1405160cc8137bebf8b42428a336f5))
* Init training script ([`1ff62ff`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/1ff62ffd59f82470b33dcc8a4f2308508078a36d))
* Intermediate refactor ([`c82fa61`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/c82fa61c984fe999b35ed86a6bc4e035cd8507bb))

### Fix
* Failing tests ([`65dc59a`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/65dc59a10e390f60917862cfd4239f306ad1059d))
* Watcher is working ([`7fd5ba1`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/7fd5ba1ca2ce5e238c8e34d9a2d0347abf4134d9))
* Type errors ([`faa43dc`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/faa43dc302e6c236cf141737b5b3aded4ad1ea0a))
* Type errors ([`45a9add`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/45a9addcec7c891dc2ec7ba11c77311c73eaa4b1))
* Feature_selection_test requires more than 1 pred col ([`7c9f0c8`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/7c9f0c8dcbf40b05f3708bfdc534bb558fd9bea2))
* Minor fixes after merge ([`4bd3e7e`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/4bd3e7ea8225ae57410920dbb729ca626421364c))
* Add data dir to synth dataset ([`bbb7dea`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/bbb7dea2c323c187542f4e88693771c1079df03c))
* Run_id is required ([`43e6ba4`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/43e6ba49d376b6bc1470ee70c09c253e6143ffc8))
* Failing tests ([`315fd3f`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/315fd3f3308c6d805cf26573a37af534c9500d41))
* Look correct lookbehind in trainer ([`a350443`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/a350443f0f8c3ce5c1e865267ed1cc9b0a9cf251))
* Various bugs in watcher ([`615b17c`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/615b17c7be9634a286666785079f1443eceb9751))
* Correct output if only 1 outcome col ([`9beedfc`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/9beedfca0bcf6281f18f096ebc5924d3f6f95030))
* Misc. fixes and refactor ([`420da4c`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/420da4cb4af70ab28a1504da861acd8b7d3c3f86))
* Failing tests ([`1999b65`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/1999b6554df20757e6eb473ebf7f539851311657))
* Remove artefact code ([`dce563e`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/dce563e61c1b9579759aaffced60a1f2d4ec5a75))
* Infer col names return list if len 1 ([`5c0503b`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/5c0503b091ab386840a9b45446e3850a52355606))
* Make watcher not archive runs that haven't finished ([`76ba99d`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/76ba99d42c2ec8957642ae14615d19543dfa3fe3))
* Misc. minor fixes for training ([`1dbd900`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/1dbd900efae2807b1d3525c6c018c29c52e543b0))
* Add wandb_group to project struct ([`3f8ccbb`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/3f8ccbb02de841bcefe36cbe861e7924c35ff492))

### Documentation
* Typo ([`d8bd92d`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/d8bd92df0f0303ab5fa9afa40bb28148fc191f37))

## v0.11.0 (2022-10-25)
### Feature
* Added methods for feature selection ([`8c087d7`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/8c087d74d83991ae668f98470e411e86bb0471be))

### Fix
* Errors introduce on merge ([`b774066`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/b77406679235bb61b362d6983045b0513edacec8))

## v0.10.0 (2022-10-20)
### Feature
* Script to run (multirun) training and wandb watcher simultaneously ([`b8954bf`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/b8954bf85b5814b304aa56efd9e161b53bc93f55))
* Sort performance before logging and try uploading again next time if no auc ([`ba70ea8`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/ba70ea890857b24571589529847f54828b89a4cb))
* Wandb watcher for asynchronous uploading of runs to wandb. ([`f6cae88`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/f6cae888080d730548a9bb3df15489ea1ff28921))

### Fix
* Make `metric_by_time_to_diagnosis` and `auc_by_calender_time` plots include both scatter and line ([`3d123d8`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/3d123d8a190dd2bcc398f348f79e8fd832d42b71))
* Correct name of wandb run ([`4821b99`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/4821b993eecb35ccd351a6c5fe467e39bb1b8a11))
* Save model predictions on overtaci ([`9b65f6f`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/9b65f6f5c1b2830a438870dc095165b48946b0ff))

## v0.9.0 (2022-10-19)
### Feature
* Check for meaningful lookbehind_combination ([`297f7c4`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/297f7c41b87520c1044944dabc3aa4805cadb3c3))

### Fix
* Error caused by merge ([`b77679d`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/b77679d45d6b8895db776f29fcb8cb2d0d5158f6))
* Convert to sets in check ([`722a282`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/722a282cb9d46498cead7dbd04006ad59b8162d2))
* Convert omegaconf types to generics ([`c85e2d5`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/c85e2d52f650192478e497c866e84c872272a53d))

## v0.8.0 (2022-10-19)
### Feature
* Log how many rows and cols are dropped ([`9bd5720`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/9bd572008b250e7ba181f5436dbd8fe9b92e9e13))

## v0.7.0 (2022-10-18)
### Feature
* Don't allow date_bins that are outside min_look_direction ([`09670ab`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/09670ab5654d01d7c9154c693791a83adf2851e2))

## v0.6.2 (2022-10-18)
### Fix
* Incorrect order of y-labels on sens heatmap ([`554ddf8`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/554ddf817e99748f6c999d02f38f490ba2c46430))

## v0.6.1 (2022-10-18)
### Fix
* Failing tests ([`6060f0f`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/6060f0f86d507fe1d29500e25da6dc465e6713d8))

## v0.6.0 (2022-10-17)
### Feature
* Remove columns with lookbehind > min_lookbehind_days ([`c4b1611`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/c4b16118990a3883d28c7d2a7c10afba2a9e6429))

### Fix
* Incorrect handling of null lookdirection vals ([`cac8838`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/cac8838001ee287bac6d4cd0d589e9852891e288))
* Allow null specification for min_lookdirection ([`d94526d`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/d94526d5a4510eac35d30a5ada7b7cfc3bfca14d))

### Documentation
* Update docstring from review ([`fce53d5`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/fce53d5f8dd0ecbe5070f70b9a3146a8d5843cd1))

## v0.5.0 (2022-10-14)
### Feature
* Default group by day ([`7249388`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/7249388cb0937e1f04080c585511b354ce58a48e))

### Fix
* More robust saving eval data ([`58c801f`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/58c801f992deb97c5164f5c95f8d32b7ce0ac00d))
* Train on GPU ([`4d07209`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/4d07209b2d61ca14e2c6a07d5a36e4556e188db3))
* Even more robust naming, write to parquet instead of pickle ([`4e55454`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/4e554544d5507f0ab70a53efd71ca65b49f0426e))
* Enable pytest on non-windows ([`9beb331`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/9beb3312f4034a86682cea722a212479384ce098))
* Disable wand in integration test for now ([`0b3b264`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/0b3b2641f33daa079be9957b346903809da957ad))
* Can't only run pytest on darwin, not sure what to do ([`ddc1638`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/ddc163882c5a9b29204bcca602f0f6bbb8c9e440))
* Run wandb when doing integration testing ([`cec3678`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/cec367838fdf6720e247d71653389ad81717881f))
* Check run exists before using for file locatin ([`63b5065`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/63b5065f30c019774f2543f3b2003e3ee981e162))
* Missing imports ([`4e6e0a0`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/4e6e0a0aa417f5db2396c485488bbbf4925a262b))
* Make eval save naming more robust ([`51eccc9`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/51eccc99bebeebb1ec09ea83f5cea5d5a4f98177))
* Missing imports ([`37a1b15`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/37a1b15150a892abe9c51cf71de89ef52d5194b8))
* F1 to recall ([`9e31f1f`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/9e31f1f943b0ce242f5e7bec6396fb211efb545a))
* Misc. changes to configs ([`3b22c4e`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/3b22c4ea1b97f259d19d7aaf242e6e669a76cca0))
* More robust max file length handling ([`b14a424`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/b14a424312e9b9b2651bcb1135b52929d13f9d08))

## v0.4.3 (2022-10-12)
### Fix
* F1 plot only scatter ([`510746a`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/510746ae8cf6c938fa41773df6d3c1dd26e59c78))

## v0.4.2 (2022-10-10)
### Fix
* Remove booster which requires different args from xgboost ([`65812d9`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/65812d95e09d490e07225d69475988f0cc9e7113))
* Make saving resistant to long cfgs ([`18a5e18`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/18a5e18ae81a1d6ce908f6a0d29808b937879281))
* Add today's date as default group ([`6cb876f`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/6cb876f7ecffaff55c70123711063e25e65d1f81))
* Binning by year to avoid bins with only one class ([`ebc3f55`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/ebc3f55a970ad1c0d66974961e8ae8516ae3bacc))

## v0.4.1 (2022-10-10)
### Fix
* Rename integration tests ([`0e30f40`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/0e30f4007960bf17f762265c6e167daae2869e08))
* Remove dummy assert ([`5603007`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/5603007e0e5d2089dceb4c4bdb0b020f068aa4dd))
* Run on all files on push and autoinstall push ([`a446418`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/a446418f2578efe10cdf1429d671990ad7c6cf88))

## v0.4.0 (2022-10-08)
### Feature
* Add t2d_parquet data source ([`beae0d4`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/beae0d4476d0b706f2a0465a48a2cc2da137b514))

### Fix
* Improvements from review ([`f65bdfa`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/f65bdfa2aa3f7729ed29ddf4ba843ba5cf6370ee))
* Type mismatch causing failing test ([`3f494a7`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/3f494a722c2eb61f3598c97dbe9a30c3e4305a87))
* Failing tests ([`a4c625b`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/a4c625ba7536769e6f85dab804c19d6cfed18f99))
* Minor linitng ([`42a6399`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/42a6399cd2e22320ba6913c5bbbf59e313b0aa20))
* Refactor and add min_date to data ([`c6e5052`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/c6e5052667e803fc06e2f3892c3d5cdf960a2c6c))
* Invalid key in crossvalidation ([`ba490e1`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/ba490e17a63ef2cdb162792980d11b091366b87c))

## v0.3.4 (2022-10-07)
### Fix
* Remove calls to altair ([`37f95ca`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/37f95ca3db7a42d422623a597e436dc4603e1a68))
* Move plt close ([`aacd2cd`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/aacd2cdadca5e298d95287c84fe0a2e48aa79356))
* Add SAVE_DIR to sens by time ([`a7d431c`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/a7d431ca77b9997c007d207929699d0d6806eb91))
* Fix bug with None lookbehind_distance ([`06fc661`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/06fc6617766f048b0b8be1f87e8f6f083cfe947d))
* Change sens_over_time plot to matplotlib ([`92cedc0`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/92cedc020ad658096c600487994acbb9a9e19b9c))
* Make dir before saving plots ([`d340149`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/d34014984f150bba77c3caa268bf60855f8e6e89))
* Change basic bar plot to matplotlib. Closes Plot bar chart in matplotlib #217 ([`f62b70d`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/f62b70d540183a456646c2b377ae74cc0c572b96))

## v0.3.3 (2022-10-06)
### Fix
* Add balanced accuracy to `model_performance` ([`142e1e9`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/142e1e95c879869f4eb8efbff940d06469d91b64))

## v0.3.2 (2022-10-04)
### Fix
* Logging to wandb on Overtaci. ([`92efa58`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/92efa58fe16f8c04de178fdd6c83dfcb8c87956d))

## v0.3.1 (2022-10-04)
### Fix
* More linting, enable pylint pre-commit hook ([`03a0be1`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/03a0be1772737379cdcb2088934f6e7666d54013))
* Some extra linting ([`de621e8`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/de621e8bbdb22addae980822f0926cfd733d1e3a))

## v0.3.0 (2022-10-03)
### Feature
* Try decomposing actions ([`316e643`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/316e64335ee1b14aaf1c0a02fc7aad474af7a09a))

### Fix
* Specify shell ([`3574577`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/35745779716151b6e3f90e5a1b93023a27943603))
* Another attempt ([`612e88e`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/612e88e92e592bcb7145ac12159161456b9cc6f4))
* Typo in action (input -> inputs) ([`e668bad`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/e668badddeb80155da0bcf26c9bd5e155b3237bb))
* Checkout repo before attempting to use action ([`29c5f92`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/29c5f9289453b09130673d1348affd36efbf63f8))
* Typo on CI ([`9774d6e`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/9774d6e10f1dd28f00745accf23ec6902eef35db))

## v0.2.0 (2022-09-30)
### Feature
* First CI release ([`1b5f5a9`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/1b5f5a91e89826ba1c530c82570eb4465008a506))

### Fix
* CI didn't run; match version with latest release ([`d14a05c`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/d14a05c4e32bb6fdc4ce7195d3a932105ef258ca))

## v0.1.6 (2022-09-30)
### Fix
* Version bump shouldn't invalidate cache in PRs ([`2cbd2b7`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/2cbd2b7c160dbf86c385d252ccf9b415f78e73fe))

## v0.1.5 (2022-09-30)
### Fix
* Only run automerge for dependabot PRs ([`4638a64`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/4638a64c4b7f59152483eb91b41730ea51aa43cf))

## v0.1.4 (2022-09-30)
### Fix
* Automerge with MB PAT ([`e708d64`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/e708d6479287c76133c2becc13d43b84b91d0abe))

## v0.1.3 (2022-09-29)
### Fix
* Add PAT to checkout for it to persist for python semantic release ([`3af0ae6`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/3af0ae6c4303d69235acfd0f4f929406068210d4))

## v0.1.2 (2022-09-29)
### Fix
* Bump version after more lenient branch protections v7 ([`05d04b4`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/05d04b4a4ed8667c90ecc3b6ace566a571a4b70d))
* Bump version after more lenient branch protections v6 ([`55c3acf`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/55c3acff60c892dd6b3f32ffeb0539f6eddd87a0))
* Bump version after more lenient branch protections v5 ([`a90f4b0`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/a90f4b0e42b8458c8db1728fb4925dc9edd08b13))
* Bump version after more lenient branch protections v4 ([`1aa2dd0`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/1aa2dd07f7103f2573eb9991d0bab5194c4fa253))
* Typo in release.yml ([`0c7d793`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/0c7d7938672e25b992f3d3d358098699a66eb14b))
* Try using personal access token to override ([`6b1170d`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/6b1170de1fcd04958afbb21fecbfd631e8e73517))

## v0.1.1 (2022-09-29)
### Fix
* Bump version after more lenient branch protections v4 ([`c231bc3`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/c231bc3dda115062ce9ad94840bf36e6a8a1d395))
* Bump version after more lenient branch protections v4 ([`3e78068`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/3e7806880598722b7a9d5200e649aa0aca7d1984))
* Bump version after even more lenient branch protections ([`fe07436`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/fe07436759b1a03bdc7c9eb738506e5f674c3d0c))
* Bump version after more lenient branch protections ([`4b90e57`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/4b90e57e0642b5d92d3359fbe68a5bdefd827468))
* Bump release version ([`8fcb3f8`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/8fcb3f8703ea57697e5f89d4aaf799aec3559ff6))
* Bump release version ([`af89cdd`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/af89cdd96a90db3305f0087cd2e8347cc3b1821d))

## v0.1.0 (2022-09-29)
### Feature
* Add binary threshold option for `model_performance` ([`528d3a0`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/528d3a0de432d607bd9528911b9cf877818aaa71))
* Add MB PAT for auto-approval ([`d933498`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/d9334982b7431d3ae887ef4fef8aacdd53ceb8de))
* Add automerge for dependabot PRs ([`5b1c962`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/5b1c9621152b933c33dc14d2f8a389b4d644bacd))
* Migrate `model_performance` to T2D ([`a161d30`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/a161d300a57c3566f715ddb67f66e87d1c683568))
* Add ranges to pr_template and dependencies ([`74124bc`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/74124bcc7c753b7b5a34f0e4688dc196c574ea1a))
* Add dependabot ([`a8e3c08`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/a8e3c08be4ff42485d347dbb279ec725741e0cb6))
* Add script to train (multiple) models, sync the best to wandb, and move wandb runs to an archive folder ([`6cb63db`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/6cb63db1d1d96ebe22692001d02d0ea868016ffb))
* Add PROJECT_ROOT_DIR ([`e915a19`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/e915a19d9f29ff13e3916c9c0be7696eaa63be86))
* Remove stop-on-concurrency ([`38f9b81`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/38f9b81fe0026420614f55e41a2ffce02c05294b))
* Invalidate cache for $HOME as well ([`bf4afde`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/bf4afdedc964d9f58dce45094d2f75ea381e0724))
* Make WandB log to disk then upload in batch ([`f5f6121`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/f5f6121f13a172a541edcc790c362c355f4592ba))
* Example for extra evaluation from model predictions ([`547fa8d`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/547fa8db426f5e434ff34c031f9c5cf215e5ee22))
* Save model evaluation to disk ([`4ff1fea`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/4ff1fead9413f7bb917ced9150be48635f2609f0))
* Update loader to take csv ([`1050b5e`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/1050b5e29a0ac46b301578d6749d64670ec5decc))
* Update dependencies ([`1a1b459`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/1a1b459e153d48175ed298e349df962244ed4047))
* Log feature importance table ([`f783d2a`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/f783d2a443495aa8c50d8906b4be259d5cdae065))
* Feature importance for ebm ([`cca0f6d`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/cca0f6dfcb93e9ead5d2f67359317da636486d87))
* Add mean warning days ([`5a61611`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/5a61611ca4be6974b1942c2735b85deb332ccadc))
* Split testing_project into overtaci and integration for github actions ([`d1a3221`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/d1a3221c696d1da5e952a9ee8a1853b8321af9f1))
* Add plots to evaluate_model ([`fe627f6`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/fe627f65a2a1fb7b261fbf483938cf1866daf26f))
* Check if we can skip poetry install ([`8310cdb`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/8310cdb41e0f7e4900e29efaeb8cf7e9b61f9b6c))
* Parralelise pytest on github actions only ([`c5ce87b`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/c5ce87b581d9eafd9425a1469193f041f1c57912))
* Integration testing conf independent of overtaci_testing ([`24a22b3`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/24a22b30cf17a56d78c1666b975ecc6ef61b5e8a))
* Add tests ([`bf77f4a`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/bf77f4a385cfe7a65e776eebd05e3d771894fdd3))
* Add remaining plots and tests ([`c3789b0`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/c3789b00c00af24ee85458b4d9f0d0bf28cf794f))
* Add heatmap ([`0acf651`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/0acf651ad6a445d1b5682529c1f6c523771d3812))
* Run pytest in parallel ([`a80dcf4`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/a80dcf4e7da01ead6040fb22427d32267f8bf5e2))
* Add cache versioning ([`29d456c`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/29d456c71b0dce0ce57bd84ba465cf03d469cc4e))
* Add flag for GPU training of xgboost ([`1027250`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/102725092705d72f36b2cefe11e0907604bbdb5b))
* Even faster caching, hopefully ([`e8314b6`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/e8314b6af0166dbd993473c76f70feb0ee663d2b))
* Add more caching ([`603c3c6`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/603c3c6fb1694117ca743250e5c57cb97a174fef))
* Added naive bayes ([`a635b2c`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/a635b2c9d405172385ef08b9041d4c9d42151438))
* Added explainable boosting machine ([`5a9aafb`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/5a9aafbd6fdf505af618d9505944c901009ced05))
* Added explainable boosting machine ([`c6d8582`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/c6d8582ff130cbb3ae1f3301b329d9f8fef631a8))
* Added z-score-normalization ([`a2faa12`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/a2faa1246fff289f6dfe215d1f7a6d505b7da00f))
* Add sensitivity_by_time_to_outcome ([`0afba8a`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/0afba8a7c9169f3e4c0ae4690428dbe6bd335c8c))
* Add logistic regression ([`28a9fb6`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/28a9fb63d82bbc736c65f7ad9d1a40b09600b582))
* Added NAs to synth data generation ([`44b679d`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/44b679df7df736a8e3d807d68844dbb389033569))
* Working parallelisation 🤘 ([`f283a0b`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/f283a0b608b03ddb786c4b1e0be0872a332f7fbf))
* Add better default params for xgboost ([`eb7d2a7`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/eb7d2a7c69e144c0f05bc9159509f4de74b56dc0))
* :rocket: compose configs ([`f9783d1`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/f9783d1a1619ecc64094fa7c94256e320306caa0))
* Add confusion matrix to performance_by_threshold ([`e1cb42c`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/e1cb42c4e4697b52c857101bb4d579791883cdbb))
* True_prevalence, positive and negative rate to performance by threshold ([`2c85067`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/2c85067978f68e23791e741780a332815dab753d))
* Add timestamp and timestamp_outcome to synth_pred_data ([`2d22172`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/2d221723a66b7e53c2be91d78635e09208ef16c3))
* Add basic pipeline test ([`a62ab76`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/a62ab76693faa2b2e6436bd92b334f1722b5a670))
* Add option for CV or not ([`4b27123`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/4b271230897c860ea6eeb063b71e852454189a60))
* Performance by threshold table ([`390fcf0`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/390fcf0bf737ea51e95aec4886b7f21f2a6ca825))
* Ignore init with isort and ignore unused imports in init ([`45ad9e9`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/45ad9e9d548b61dca0280b975d2ae3d1854e6ac9))
* Log performance metrics ([`678bc37`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/678bc37cd1ff096eeaf380fa38865b118d34a29a))
* Migrate to new package structure, upgrade ([`c114d19`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/c114d1930d4a9e5a4a7d982068ac86399ca7c814))
* Migrate to new package structure, upgrade ([`cd8390a`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/cd8390a8f169385922fc5ce9acbd9903a2ad2aa9))
* Add cache to pytest ([`85e1dc4`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/85e1dc4794c48ebd38f9f03271907151d8f26ec8))

### Fix
* Renamed dependency ([`e73734e`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/e73734e37c705b100d52efc35223e1526262d2f9))
* Bug if `ModelPerformance` used without id2label ([`b928ab7`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/b928ab7085454b450256dd81bf4ba916f635d17e))
* Improvements from review ([`1d88ea0`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/1d88ea07763e1333debe35f516f3ac4d03acc556))
* Invalid placement of version key for poetry install ([`52fbebe`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/52fbebeb091e4ae18b3c34ace4503865a7944ad9))
* Update documentation of GA. ([`70442fe`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/70442fe4f68db97ac5d184ab80f170441051046c))
* Allow duplication of small utils ([`2c1b043`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/2c1b043be4e4ef57690bc04f40f3ad59926c37a2))
* Ensure cache depends on python-version ([`1052caf`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/1052caf1742f65f2075901e6756ef5404ee2224a))
* Overspecified python 3.9.0 slowing down pytest ([`aa14654`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/aa14654491e6aedf2c88fc90ea87f3c90a99bfe7))
* Typo in pytest ([`8ca7a3b`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/8ca7a3b610c8f6221b596a1568619c4c573d4fa1))
* Altair_saver requires selenium <4.3.0 ([`eb99574`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/eb99574d70eb0b81cc88a90bbcdceb80e9eda0a9))
* Minor test changes ([`a3f1f5f`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/a3f1f5fa04a5f5c95325f25c10644fe3b2cdab5f))
* Remove interpret to avoid llvmlite ([`36f1d01`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/36f1d01b18442304568bc6882f943197919e7226))
* Misspecified types ([`94e5fd4`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/94e5fd4edf7efbe5b894fdf3053dee2ef6d44208))
* Another typo in .toml, remove darglint as dep ([`df34e1b`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/df34e1b584a3d14e93d92d6c5fb97fb321b27d77))
* Typo in toml ([`57d3d02`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/57d3d02e933736090f920bd52739c101f4f5fce2))
* Spelling error in toml ([`8d67ded`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/8d67ded228faa3969404c3e19adc7df23c6466f7))
* Fix failing tests ([`f06ac16`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/f06ac168c96239f4bbfe1d23a8759cb5ae346230))
* Type in .venv sourcing ([`d154d13`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/d154d137c3338bb8a7bb91ab31c7b975dff9e401))
* Remove unix-specific github actions runs ([`e000a90`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/e000a90eaef76770fa1201b4d894142db4904da7))
* Minor fixes ([`3505e70`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/3505e70d1bdc5209bbfa6bdd7890d1f012b340e3))
* Remove mypy type ignores ([`ae66393`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/ae66393c5b98b9b5f5c8191cde8fa015c4a81a2f))
* Succesful training on overtaci ([`8e84c4b`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/8e84c4b7acb6bd00f3225bdd99007d1a557ca4e7))
* Minor changes ([`2222141`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/22221418decc9177f4eef7e04b6bd1f635c4430d))
* Failing tests ([`1c5ace1`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/1c5ace15a1aba370a5051753c8a65964238cd973))
* Clean darglint ([`41d5565`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/41d556562e84b204b7fcf4a5f57f09d7ca9eb77f))
* Rename github action ([`35f3dd5`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/35f3dd571c2b79da33d5bac49356084f7f0463ac))
* Update t2d outcome ([`0bda7bd`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/0bda7bd778da42bc206ee09b819957878ae1ef47))
* Failed import ([`885c8e2`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/885c8e25c6569f1a5802bc1a837cd427c4c4e23e))
* Wrong direction of bool ([`f8ed7f3`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/f8ed7f3e8676ba7ee74f78919f2d31a38c480f16))
* Sort feature importance plot ([`cbe81b3`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/cbe81b305b62fee9f876110eb76a773b004bf703))
* Change argsort direction ([`19d635d`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/19d635d88f022ddf5a651ce0dbb82d895b7dd4a3))
* Control n features to plot ([`0e7bc23`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/0e7bc23cff5283efbde54852231959195c923c5a))
* Skip feature_importances_ for ebm ([`be1c8d4`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/be1c8d46756e4ea029ecdcf4477fedf4709a44a8))
* Check if array exists ([`edabb56`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/edabb5675e1a4c82f244e3d275cc6cc254b6a221))
* Refer to correct test project on overtaci ([`b4210d8`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/b4210d8e3b271842792ef107d3071a9cfe799763))
* Add mean_warning_days to test ([`283a6cc`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/283a6cca3049be8b91d4735f8ce9637d2a977d1d))
* Add mean_warning_days to test ([`c2dc20f`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/c2dc20fc9faa91078d128a64f02d17becf0e33f7))
* Incorrect calculation of warning days per false positive ([`235e404`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/235e404120666dacccd7f24ed74832a5a1ae85cd))
* Add chromedriver to GH actions ([`ce9478d`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/ce9478d916cf64a417a6d712c39a2aff22dcaca1))
* Update tests ([`3bc6873`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/3bc68735b93497cced9dc05f196a692cce80d519))
* Round decimals to percent ([`cdb3346`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/cdb3346a977a6b2f3a57e106c169d3c300ea64f3))
* Remove pytest.ini ([`d041a16`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/d041a1664e9643145c54d9dcf6aa9af8b7356fb4))
* Return nan if only 1 class in AUC calculation ([`17c1380`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/17c1380925e5404902b060fb1e2243fb393a6614))
* Scale y-labels to whole numbers, e.g. 0.5 -> 50% ([`9d3ca8f`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/9d3ca8f1ccc1870d15bf2d0fd973281a83257010))
* Config_naming ([`874a564`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/874a56497e5c686bfe040efbd8f03b80ab969201))
* We must install poetry ([`f2474e6`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/f2474e6537b64a9d87020091ed41cf498aedbd8c))
* Enable newline interpretation for pytest.ini ([`25f0026`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/25f00261624e516f7392de42df1bb155a082d750))
* Try another way of parralising ([`5fcec4d`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/5fcec4d4a0a84845061b034ca3ee4a71d09bc147))
* Remove plot saving ([`5c6819c`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/5c6819c289593163dad1137a89f309ccc197a3e4))
* Tentative fix of warning_days ([`1bf99e5`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/1bf99e558cd5615a7ffe8d21da51608f98901510))
* Remove pytest parallelisation, slows down locally ([`05cfc37`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/05cfc37bb56ed2631d3af1088ae127eff103ddea))
* Titling of legend ([`884e08d`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/884e08d8365c2cce161eb0b6989c9e3a70448ab3))
* Cleanup ([`c7d1fb0`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/c7d1fb0647d924cbda40b4387c732ac1eb029a64))
* Set known_third_party for wandb ([`4cc0ec7`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/4cc0ec7b68cdd6d7813ce14e7792122c1d6a410c))
* Caching ([`b587b14`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/b587b1473df6bf531c2536a66cc63bdea6ca2485))
* Handle ebm on arm ([`dccd3ce`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/dccd3ce3d3804b4255a3c9fd636f8ea165effc8a))
* Apply recommendations ([`306bdb9`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/306bdb93b91cadd90bf9a3359df08340ee54af66))
* Apply recommendations ([`c85fb55`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/c85fb55aaf305817824eba13352399513973d6a9))
* Create figures folder if it doesn't exist ([`0a86e29`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/0a86e29446c32b73027f5f4b9fee9bdb0d5de865))
* Add folder to save html to in tests ([`4323e83`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/4323e83557e73de6af87ec6837208ed9b94a7471))
* Downgrade selenium since altair uses deprecated api ([`2ba94a2`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/2ba94a2d14d52a7bac1452da99f4b4df304294b4))
* Fixed bug when there is NAs in the pred_timestamps ([`2df71fe`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/2df71fe438537742cb0e71e161914e641f829c46))
* Update config for test ([`5f16eef`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/5f16eef882d228a982047b826099f2d5cc9ff389))
* Simplify configs ([`f9cee28`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/f9cee28f326fcff854275b2c0bf56c2b02de0ab8))
* Hyperparameters sweep works ([`5ac5f1a`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/5ac5f1a17ce3d7661ffa9f019b84707f7fcb5ba7))
* Working example ([`9f66d23`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/9f66d23e2cba94961deccd72249d92b52bc05449))
* First working example ([`0e52681`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/0e52681a1ed0e8e80348e9eb4d710b0dc9fccc3e))
* Wandb testing with dryrun ([`142b2cc`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/142b2ccbbe1160109b619918e3d9e41b26e208b9))
* Remove random gitignore strings ([`6c42c23`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/6c42c230af21251bb5e8f072395c6d7e2e150936))
* Random gitnogre string ([`c289822`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/c289822a4d958d75848dd78b70f8e07c89a45a57))
* Dependencies fixed ([`81d2740`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/81d274075b7c5f32e2d95f41c0a5532a23c74968))
* Set upper bounds for packages ([`b91026e`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/b91026ee7827acd0ba0629849214855d7b6fc0fe))
* Add support for wandb_run dryrun ([`9e227cd`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/9e227cd06e90cb84a8f9247d35dd0c1ba1874b30))
* Tests not passing ([`af9ecb9`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/af9ecb988b1bcba1b2a279d8b820bd29d1ed13a2))
* Remove " in test_basic_pipeline.yaml ([`c1b5b22`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/c1b5b223a247017135597a18571cd7945dab4340))
* Remove " from yaml config ([`8693d14`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/8693d1452dc1c4222fc131e4b41847952a8099de))
* Failing tests ([`21e3397`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/21e339723266121ba9f7c6faba1fdc0ed27ba295))
* Bugs in performance_by_threshold ([`8ddd71d`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/8ddd71db15d61610c41053751f66789f91c5ae67))
* Bugs ([`3506541`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/3506541fe154a85dd715929086d7719d7e105a5b))
* Update synth_prediction_data ([`3808757`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/3808757a4f81b24d49b0bf0693d272a4bec62deb))
* Generalise flatten_nested_dict ([`521baf0`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/521baf0b3f21f041025da770340b621fcae5a30c))
* Check caching ([`afb16e9`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/afb16e980814de44bf1bbbf1d49d5bb5c12e4dbb))
* Ga pytest not caching dependencies ([`fad35e9`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/fad35e9e5de75dd16c9ac7077ebc1d4e40a54ce1))
* Ignore .csv again ([`6a9a764`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/6a9a7645bee7c0ca50023c7556c3e48105e6da1b))
* Add fake data files ([`4fbcab9`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/4fbcab912efed3772d9ea178283419c668c81cd8))
* Tests not passing ([`b78dc20`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/b78dc201b574108d1481a1c95faff3a4e93db995))
* Failing tests due to path rename ([`8407307`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/8407307dad4e6e412df7c7de9fddf37dba1edf15))
* First train with synth data ([`a4195ee`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/a4195ee31f1ffec758cc0dc9e9ba9402df2db033))
* Run on all examples ([`7795e06`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/7795e0623773838571fb07ac10d93a8d1b4ea32e))
* Working version! ([`5f803f4`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/5f803f43e5702c425db0396e8986f306151ac7b8))
* Test MVP ([`d92b6be`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/d92b6be09b6d86993e9179c054d248d950365cec))
* Add misc corrections ([`a17fce7`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/a17fce71940307cf34b1c3002cac23ed1b54e375))
* Basic version working ([`b1d5e10`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/b1d5e101a40722528214cbcb30b983554c9349f7))
* Misc ([`fa6a8ba`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/fa6a8baabb93953879f5492536443b46050290eb))
* Use pyproject.toml for poetry github action caching ([`db1e542`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/db1e54280c0b309cdc33cc4db5e481cc8d951483))
* Update dependencies ([`41cff75`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/41cff754ebebd9172298922fa7816251d640bc97))
* Update poetry.lock ([`4cd40bb`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/4cd40bbdb40a8f3b76cd2d83d2af80b64bd76243))
* Improve from Lasse's comments ([`528864d`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/528864df3264d4014d629f54ed86f6f369ff61c6))
* Keyerror on test ([`83bc031`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/83bc031909dfc05672b2e196db83526678a40e48))
* Remove ambiguous comment ([`a4413bc`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/a4413bc25e00e46e6a178b6ec4be86170adedf62))
* Update pytest version ([`44eb8d5`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/44eb8d5359e4bbf1e0a3a403b8353780a7aca1b2))

### Documentation
* Add missing docstrings ([`664c8d3`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/664c8d34d9aef246c44f4b325405bf65ebd45d33))
* Some info on feature importance ([`a22a325`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/a22a325f1c2a739189c1250411654035202b8f6c))
* Expand local chromedriver set up ([`348cb70`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/348cb7043f7e8e1e23a3d9814e06e63684baf22b))
* Fix links to chromedriver ([`4032a79`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/4032a791daeec84e03d9200888fc1badd4baae3e))
* Installing chromedriver ([`7c40ea4`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/7c40ea4d7af9090e8bf7a28b7bdb6bb620d2b13d))
* Update docstring ([`8231659`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/8231659041ad7b3883edb97ac41112ef5dd687b0))
* Updated train model introduction ([`6fe1353`](https://github.com/Aarhus-Psychiatry-Research/psycop-t2d/commit/6fe1353918ece9a152ab00f40c90e089dd7ca4d2))
