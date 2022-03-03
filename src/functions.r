event_start_date_str <- "2014-01-01"

str_contains_diab_diag <- function(str) {
  diabetes_regex_pattern <- "(:DE1[1-5].*)|(:DE16[0-2].*)|(:DO24.*)|(:DT383A.*)|(:DM142.*)|(:DG590.*)|(:DG632*)|(:DH280.*)|(:DH334.*)|(:DH360.*)|(:DH450.*)|(:DN083.*)"

  if (isTRUE(str_detect(str, diabetes_regex_pattern))) {
    return(TRUE)
  }
  
  return(FALSE)
}

keep_only_first_diabetes_by_diag <- function(df, date_col_string) {
  diag_fun_vec <- Vectorize(str_contains_diab_diag)

  df %>% 
    filter(diag_fun_vec((diagnosegruppestreng))) %>% 
    group_by(dw_ek_borger) %>% 
    filter(date_col_string == min(date_col_string)) %>% 
    filter(row_number() == 1) %>% 
    ungroup()
}