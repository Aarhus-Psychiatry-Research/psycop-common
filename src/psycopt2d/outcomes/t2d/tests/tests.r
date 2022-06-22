
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
output_df <- keep_only_first_t2d_by_diag(test_df, "datotid_lpr3kontaktstart")

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

test_window_gen_df <- tribble(
  ~years_from_visit_to_t2d, ~years_to_end_of_follow_up, ~dw_ek_borger,
  1, 1, 1,
  1, 2, 2
)

output_df_window <- mutate(test_window_gen_df, window_1 = visit_can_generate_prediction_vecced(years_from_visit_to_t2d, years_from_visit_to_t2d, 1))

test_window_grouped_df <- tribble(
  ~dw_ek_borger, ~window_1, ~window_2,
  1, 1, 0,
  1, 0, 0,
  1, 0, 1,
  2, 0, 0,
  2, 1, 0
)

df_out_window_group <- test_window_grouped_df %>% 
  group_by(dw_ek_borger) %>% 
  summarise(across(starts_with("window"), max, .names = "{.col}"))