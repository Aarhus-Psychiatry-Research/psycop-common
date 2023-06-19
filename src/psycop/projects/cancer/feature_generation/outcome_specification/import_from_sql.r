library("pacman")
p_load(DBI, dplyr, lubridate)
options(lubridate.fasttime = TRUE) # Uses the fasttime implementation of POSIXct conversion, speeds up date conversion ~10x

#' Get all contents from SQL view
#' @param fct The fact name
#'
#' @return A dataframe of the fct
#'
get_fct <- function(fct) {
    conn <- DBI::dbConnect(
        odbc::odbc(),
        Driver = "SQL Server",
        Server = "BI-DPA-PROD",
        database = "USR_PS_Forsk",
        Trusted_Connection = "TRUE"
    )

    return(dbGetQuery(conn, paste0("SELECT * FROM [fct].", fct)))
}

#' Fix typical issues with the SQL import we get
#' - Convert dates from chr to POSIXct
#' - Remove leading Ds from adiagnosis (e.g. DF30 to F30)
#' - Lowercase all column names
#'
#' @param df (df)
#' @param convert_dates_to_posixct (bool)
#'
#' @return A cleaned dataframe
#'
#' Example run
#' df_data <- get_fct("FOR_LPR3kontakter_psyk_somatik_inkl_2021") %>%
#' format_sql_import()

format_sql_import <- function(df) {
    df <- df %>%
        rename_with(tolower) %>%
        mutate(across(matches(".*diagnosekode.*"), ~ substr(.x, 2, 99))) %>% # Remove leading Ds
        mutate(across(matches("datotid.+"), ~ ymd_hms(.x))) %>% # Convert typical datetime columns to POSIXct
        mutate(across(matches("(.)+dato"), ~ ymd(.x))) # Convert typical date strings to date

    return(df)
}