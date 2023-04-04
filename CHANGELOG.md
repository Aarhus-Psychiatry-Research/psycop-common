# Changelog

<!--next-version-placeholder-->

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
* Move values_df resolution to _AnySpec object ([`714e83f`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/714e83fd3722b298cdd256b06915659ca7a34259))
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
