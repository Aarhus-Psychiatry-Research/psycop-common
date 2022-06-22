library(DBI)
library(dplyr)
library(lubridate)
library(ggplot2)
library(stringr)
library(tidyr)
library(tidyverse)
library(ggdist)
library(skimr)
library(patchwork)

conn <- DBI::dbConnect(
  odbc::odbc(),
  Driver = "SQL Server",
  Server = "BI-DPA-PROD",
  database = "USR_PS_Forsk",
  Trusted_Connection = "TRUE"
)

##################
# Planned visits #
##################
df_planned_visits_raw <- dbGetQuery(conn, "SELECT * FROM [fct].FOR_besoeg_fysiske_fremmoeder") %>% 
  rename_with(tolower)

df_planned_psych_visits <- df_planned_visits_raw %>% 
  filter(substr(shakafskode_besoeg, 1, 4) == "6600") %>% 
  filter(psykambbesoeg == 1) %>% 
  select(dw_ek_borger, datotid_start) %>% 
  arrange(datotid_start)

df_p_samp <- df_planned_psych_visits %>% 
  filter(dw_ek_borger == 31) %>% 
  arrange(datotid_start)

# Iterate over planned visits to only keep those, that are not within 3 months from last prediction
drop_within_3_months_from_prediction <- function(df) {
  # Only takes as an input a dataframe that is already sorted by date (!!!)
  current_CPR <- 0
  patient_i <- 0
  last_selected_date <- 0
  indeces_to_drop <- c()
  
  for (i in 1:nrow(df)) {
    # print(str_c("Row_CPR, Current CPR: ", df$dw_ek_borger[i], ", ", current_CPR))
    
    if (df$dw_ek_borger[i] != current_CPR) { # Handle switching to new person
      current_CPR = df$dw_ek_borger[i]
      last_selected_date = ymd_hms(df$datotid_start[i])
      
      if (patient_i %% 100 == 0 ) {
        print(str_c("Processing patient nr. ", patient_i))
      }
      
      patient_i <- patient_i + 1
      
      next()
    }
    
    if (df$dw_ek_borger[i] == current_CPR) { # Handle comparison of current visit to previous selected date
      if (ymd_hms(df$datotid_start[i]) < (as.Date(last_selected_date) + 90)) {
        indeces_to_drop <- c(indeces_to_drop, i)
      } else {
        last_selected_date <- df$datotid_start[i]
      }
    }
  }
  
  return(df %>% slice(-indeces_to_drop))
}

df_planned_with_3m_spacing <- drop_within_3_months_from_prediction(df_planned_psych_visits)



#######
# Age #
#######
df_demo_raw <- dbGetQuery(conn, "SELECT * FROM [fct].FOR_kohorte_demografi") %>% 
  rename_with(tolower)

df_demo <- df_demo_raw %>% 
  select(foedselsdato, dw_ek_borger) %>% 
  mutate(foedselsdato = ymd(foedselsdato))

###############
# First psych #
###############
df_psyk_raw <- dbGetQuery(conn, "SELECT * FROM [fct].FOR_besoeg_fysiske_fremmoeder") %>% 
  rename_with(tolower)

df_first_p <- df_psyk_raw %>% 
  select(dw_ek_borger, datotid_start) %>% 
  group_by(dw_ek_borger) %>% 
  arrange(datotid_start, .by_group=TRUE) %>% 
  filter(row_number() == 1) %>% 
  rename(datotid_f_psych = datotid_start)


###############
# T2D samples #
###############
# Raw
df_hba1c_raw <- dbGetQuery(conn, "SELECT * FROM [fct].FOR_LABKA_NPU27300_HbA1c") %>% 
  rename_with(tolower)

df_maybe_t2d <- df_hba1c_raw %>% 
  select(datotid_godkendtsvar, svar, dw_ek_borger) %>% 
  mutate(svar = as.numeric(svar)) %>% 
  filter(svar > 47) %>% 
  select(-svar) %>% #Remove incidences that are before first psych contact
  left_join(df_first_p) %>% 
  filter(datotid_f_psych < datotid_godkendtsvar) %>% 
  group_by(dw_ek_borger) %>% 
  arrange(datotid_godkendtsvar, .by_group = TRUE) %>% # Keep only first row
  filter(row_number() == 1) %>% 
  rename(datotid_maybe_t2d = datotid_godkendtsvar) %>%
  mutate(datotid_maybe_t2d = ymd_hms(datotid_maybe_t2d)) 


df_probably_t2d <- df_hba1c_raw %>% 
  select(datotid_godkendtsvar, svar, dw_ek_borger) %>% 
  mutate(svar = as.numeric(svar)) %>% 
  filter(is.na(svar) == FALSE) %>% #Remove incidences that are before first psych contact
  left_join(df_first_p) %>% 
  filter(datotid_f_psych < datotid_godkendtsvar) %>% 
  group_by(dw_ek_borger) %>% # Check if first HbA1c was normal
  arrange(datotid_godkendtsvar, .by_group=TRUE) %>% 
  mutate(first_hba1c_normal = svar[1] < 48) %>% 
  filter(svar > 47 & first_hba1c_normal == TRUE) %>% 
  group_by(dw_ek_borger) %>% 
  arrange(datotid_godkendtsvar, .by_group = TRUE) %>% 
  filter(row_number() == 1) %>% # Keep only first match
  select(datotid_godkendtsvar) %>% 
  rename(datotid_probably_t2d = datotid_godkendtsvar) %>% 
  mutate(datotid_probably_t2d = ymd_hms(datotid_probably_t2d))

########
# Plot #
########
setwd("E:/Users/adminmanber/Desktop/T2D")

##############
# Age at T2D #
##############
gen_plot_age_df <- function(df, outcome) {
  df_out <- df %>% 
    left_join(df_demo) %>% 
    mutate(age_at_t2d = interval(foedselsdato, {{outcome}}) / years(1))
  
  return(df_out)
}

df_plot_probably_t2d <- gen_plot_age_df(df_probably_t2d, datotid_probably_t2d)

df_plot_maybe_t2d <- gen_plot_age_df(df_maybe_t2d, datotid_maybe_t2d)

save_histogram <- function(df, x_var, filename) {
  gg <- ggplot(df, aes(x={{x_var}})) +
    geom_histogram(binwidth=1) +
    labs(
      title = filename,
      x = "Age at incident T2D (years)",
      y = "Count"
    ) +
    scale_x_continuous(
      breaks = seq(15, 100, by=5),
      limits = c(15, 100)
    )
  
  ggsave(str_c("figures/", filename, ".png"), width = 20, height = 10, dpi = 100, units = "in")
  
  gg
}

save_histogram(df_plot_probably_t2d, age_at_t2d, "age_at_first_t2d_hba1c_after_normal_hba1c")
save_histogram(df_plot_maybe_t2d, age_at_t2d, "age_at_first_t2d_hba1c")


##################################
# Time from planned visit to T2D #
##################################
gen_planned_to_event_df <- function(df_event, event_col, df_planned_visits) {
  df_out <- df_event %>% 
    inner_join(df_planned_visits) %>%
    rename(visit_start = datotid_start) %>% 
    mutate(years_since_visit = interval(visit_start, {{event_col}}) / years(1)) %>% 
    filter(years_since_visit > 0.25)
  
}

df_planned_to_probable_t2d <- gen_planned_to_event_df(df_event=df_probably_t2d, 
                                                      event_col=datotid_probably_t2d, 
                                                      df_planned_visits=df_planned_psych_visits)

df_planned_to_maybe_t2d <- gen_planned_to_event_df(df_event=df_maybe_t2d, 
                                                   event_col=datotid_maybe_t2d, 
                                                   df_planned_visits=df_planned_psych_visits)

save_time_from_visit <- function(df, x_var, filename) {
  gg <- ggplot(df, aes(x={{x_var}})) +
    scale_x_continuous(
      breaks = seq(0, 10, by=0.25),
      limits = c(0, 10)
    )
  
  hist <- gg +
    geom_histogram(
      binwidth = 0.25
    ) 
  
  box <- gg +
    geom_boxplot()
  
  combined <- hist + box + plot_layout(nrow = 2, height = c(2, 1))
  
  ggsave(str_c("figures/", filename, "_histogram.png"), width = 20, height = 20, dpi = 100, units = "in")
  
  combined
}

save_time_from_visit(df_planned_to_maybe_t2d, years_since_visit, "years_until_maybe_t2d_for_visit_histogram")
save_time_from_visit(df_planned_to_probable_t2d, years_since_visit, "years_until_probable_t2d_for_visit_histogram")
