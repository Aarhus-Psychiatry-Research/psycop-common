
library("pacman")

p_load(testthat, here, xpectr)

source(here("src", "functions.r"))

test_df <- tribble(
  ~diagnosegruppestreng, ~datotid_lpr3kontaktstart, ~dw_ek_borger,
  "A:DE14#+:ALFC3", "2021-06-30 09:00:00.0000000", 1,
  "A:DE14#+:ALFC3", "2021-05-30 09:00:00.0000000", 1,
  "A:DE14#+:ALFC3", "2021-04-30 09:00:00.0000000", 1
)

source(here("src", "functions.r"))
output_df <- keep_only_first_diabetes_by_diag(test_df, "datotid_lpr3kontaktstart")

test_that("Correct diagnosegruppe-matching",{
  # Testing column values
  expect_equal(
    output_df[["diagnosegruppestreng"]],
    "A:DE14#+:ALFC3",
    fixed = TRUE)
  expect_equal(
    output_df[["dw_ek_borger"]],
    1,
    tolerance = 1e-4)
})