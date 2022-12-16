event_start_date <- ymd("2014-01-01")

str_contains_t2d_diag <- function(str) {
  t2d_regex_pattern <- "(:DE1[1-5].*)|(:DE16[0-2].*)|(:DO24.*)|(:DT383A.*)|(:DM142.*)|(:DG590.*)|(:DG632*)|(:DH280.*)|(:DH334.*)|(:DH360.*)|(:DH450.*)|(:DN083.*)"

  if (isTRUE(str_detect(str, t2d_regex_pattern))) {
    return(TRUE)
  }
  
  return(FALSE)
}

keep_only_first_t2d_by_diag <- function(df, date_col_string) {
  str_contains_t2d_diag_vecced <- Vectorize(str_contains_t2d_diag)

  df %>% 
    filter(str_contains_t2d_diag_vecced((diagnosegruppestreng))) %>% 
    group_by(dw_ek_borger) %>% 
    filter(date_col_string == min(date_col_string)) %>% 
    filter(row_number() == 1) %>% 
    ungroup()
}

str_contains_t1d_diag <- function(str) {
  t1d_regex_pattern <- "(:DE10.*)|(:DO240.*)"

  if (isTRUE(str_detect(str, t1d_regex_pattern))) {
    return(TRUE)
  }
  
  return(FALSE)
}

keep_only_first_t1d_by_diag <- function(df, date_col_string) {
  str_contains_t1d_diag_vecced <- Vectorize(str_contains_t1d_diag)

  df %>% 
    filter(str_contains_t1d_diag_vecced((diagnosegruppestreng))) %>% 
    group_by(dw_ek_borger) %>% 
    filter(date_col_string == min(date_col_string)) %>% 
    filter(row_number() == 1) %>% 
    ungroup()
}

visit_can_generate_prediction <- function(col1, col2, window_width_years) {
 if_else(({{col1}}<years) & ({{col2}}<years), 1, 0)
}

visit_can_generate_prediction_vecced <- Vectorize(visit_can_generate_prediction)