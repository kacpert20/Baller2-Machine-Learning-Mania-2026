library(readr)
library(MASS)
library(dplyr)
# =========================================================
# 0. ŚCIEŻKI
# =========================================================
DIR_INTERIM <- "data/interim"
DIR_OUT <- file.path(DIR_INTERIM, "r_outputs")
dir.create(DIR_OUT, recursive = TRUE, showWarnings = FALSE)

# =========================================================
# 1. WCZYTANIE DANYCH
# =========================================================
m_train <- read_csv(file.path(DIR_INTERIM, "m_matchups_train_2014_2024.csv"), show_col_types = FALSE)
m_valid <- read_csv(file.path(DIR_INTERIM, "m_matchups_valid_2025.csv"), show_col_types = FALSE)
m_full  <- read_csv(file.path(DIR_INTERIM, "m_matchups_full_train_2014_2025.csv"), show_col_types = FALSE)
m_pred  <- read_csv(file.path(DIR_INTERIM, "m_matchups_stage2_2026.csv"), show_col_types = FALSE)

w_train <- read_csv(file.path(DIR_INTERIM, "w_matchups_train_2014_2024.csv"), show_col_types = FALSE)
w_valid <- read_csv(file.path(DIR_INTERIM, "w_matchups_valid_2025.csv"), show_col_types = FALSE)
w_full  <- read_csv(file.path(DIR_INTERIM, "w_matchups_full_train_2014_2025.csv"), show_col_types = FALSE)
w_pred  <- read_csv(file.path(DIR_INTERIM, "w_matchups_stage2_2026.csv"), show_col_types = FALSE)

# =========================================================
# 2. METRYKI
# =========================================================
brier_score <- function(y, p) {
  mean((p - y)^2)
}

log_loss <- function(y, p, eps = 1e-15) {
  p <- pmin(pmax(p, eps), 1 - eps)
  -mean(y * log(p) + (1 - y) * log(1 - p))
}

# =========================================================
# 3. LISTY KANDYDATÓW
#    celowo tylko sensowne, "turniejowe" i dostępne dla stage2
# =========================================================
men_candidates <- c(
  "Diff_SeedNum",
  "Diff_WinPct",
  "Diff_NetRtgAvg",
  "Diff_OffRtgAvg",
  "Diff_DefRtgAvg",
  "Diff_eFGAvg",
  "Diff_TOVPctAvg",
  "Diff_ORBPctAvg",
  "Diff_DRBPctAvg",
  "Diff_Last10WinPct",
  "Diff_Last10NetRtgAvg",
  "Diff_Last5WinPct",
  "Diff_Last5NetRtgAvg",
  "Diff_ConferenceWinPct",
  "Diff_ConfTourneyWins",
  "Diff_ConfTourneyWinPct",
  "Diff_OppWinPctAvg",
  "Diff_OppNetRtgAvg",
  "Diff_MasseyRankMean",
  "Diff_MasseyRankMedian",
  "Diff_MasseyRankBest",
  "Same_ConfAbbrev"
)

women_candidates <- c(
  "Diff_SeedNum",
  "Diff_WinPct",
  "Diff_NetRtgAvg",
  "Diff_OffRtgAvg",
  "Diff_DefRtgAvg",
  "Diff_eFGAvg",
  "Diff_TOVPctAvg",
  "Diff_ORBPctAvg",
  "Diff_DRBPctAvg",
  "Diff_Last10WinPct",
  "Diff_Last10NetRtgAvg",
  "Diff_Last5WinPct",
  "Diff_Last5NetRtgAvg",
  "Diff_ConferenceWinPct",
  "Diff_ConfTourneyWins",
  "Diff_ConfTourneyWinPct",
  "Diff_OppWinPctAvg",
  "Diff_OppNetRtgAvg",
  "Same_ConfAbbrev"
)

# =========================================================
# 4. FUNKCJE POMOCNICZE
# =========================================================
prepare_frames <- function(train_df, valid_df, pred_df, candidate_vars) {
  common_vars <- Reduce(
    intersect,
    list(candidate_vars, names(train_df), names(valid_df), names(pred_df))
  )
  
  # tylko numeryczne / binarne
  common_vars <- common_vars[
    sapply(train_df[common_vars], function(x) is.numeric(x) || is.integer(x))
  ]
  
  train_x <- train_df %>% dplyr::select(Target, dplyr::all_of(common_vars))
  valid_x <- valid_df %>% dplyr::select(Target, dplyr::all_of(common_vars))
  pred_x  <- pred_df  %>% dplyr::select(ID, dplyr::all_of(common_vars))
  # imputacja medianą z train
  medians <- list()
  for (v in common_vars) {
    med <- median(train_x[[v]], na.rm = TRUE)
    if (!is.finite(med)) med <- 0
    medians[[v]] <- med
    
    train_x[[v]][is.na(train_x[[v]])] <- med
    valid_x[[v]][is.na(valid_x[[v]])] <- med
    pred_x[[v]][is.na(pred_x[[v]])]   <- med
  }
  
  # usuwamy zmienne stałe
  keep_vars <- common_vars[
    sapply(train_x[common_vars], function(x) dplyr::n_distinct(x) > 1)
  ]
  train_x <- train_x %>% dplyr::select(Target, dplyr::all_of(keep_vars))
  valid_x <- valid_x %>% dplyr::select(Target, dplyr::all_of(keep_vars))
  pred_x  <- pred_x  %>% dplyr::select(ID, dplyr::all_of(keep_vars))
  
  list(
    train = train_x,
    valid = valid_x,
    pred  = pred_x,
    vars = keep_vars,
    medians = medians
  )
}

run_stepwise_glm <- function(train_df, valid_df, pred_df, candidate_vars, label = "MODEL") {
  prep <- prepare_frames(train_df, valid_df, pred_df, candidate_vars)
  
  upper_formula <- reformulate(prep$vars, response = "Target")
  
  fit_null <- glm(
    Target ~ 1,
    data = prep$train,
    family = binomial(),
    control = glm.control(maxit = 100)
  )
  
  fit_step <- stepAIC(
    fit_null,
    scope = list(lower = ~1, upper = upper_formula),
    direction = "both",
    trace = FALSE
  )
  
  # Cook's distance
  cooks_d <- cooks.distance(fit_step)
  cook_thr <- 4 / nrow(prep$train)
  keep_idx <- is.na(cooks_d) | cooks_d <= cook_thr
  
  fit_final <- glm(
    formula(fit_step),
    data = prep$train[keep_idx, , drop = FALSE],
    family = binomial(),
    control = glm.control(maxit = 100)
  )
  
  p_valid <- predict(fit_final, newdata = prep$valid, type = "response")
  p_valid <- pmin(pmax(p_valid, 1e-6), 1 - 1e-6)
  
  metrics <- tibble(
    model = label,
    n_train = nrow(prep$train),
    n_train_after_cook = sum(keep_idx),
    n_valid = nrow(prep$valid),
    n_candidates = length(prep$vars),
    n_selected = length(all.vars(formula(fit_final))) - 1,
    brier_valid = brier_score(prep$valid$Target, p_valid),
    logloss_valid = log_loss(prep$valid$Target, p_valid)
  )
  
  cat("\n=====================================================\n")
  cat(label, "\n")
  cat("=====================================================\n")
  print(metrics)
  cat("\nWybrane zmienne:\n")
  print(all.vars(formula(fit_final))[-1])
  
  list(
    fit = fit_final,
    metrics = metrics,
    selected_vars = all.vars(formula(fit_final))[-1],
    cook_threshold = cook_thr,
    keep_idx = keep_idx
  )
}

refit_on_full_train <- function(full_df, pred_df, selected_vars, label = "FINAL_MODEL") {
  common_vars <- Reduce(
    intersect,
    list(selected_vars, names(full_df), names(pred_df))
  )
  
  full_x <- full_df %>% dplyr::select(Target, dplyr::all_of(common_vars))
  pred_x <- pred_df %>% dplyr::select(ID, dplyr::all_of(common_vars))
  # imputacja medianą z full train
  for (v in common_vars) {
    med <- median(full_x[[v]], na.rm = TRUE)
    if (!is.finite(med)) med <- 0
    
    full_x[[v]][is.na(full_x[[v]])] <- med
    pred_x[[v]][is.na(pred_x[[v]])] <- med
  }
  
  # usuwamy ewentualne stałe
  keep_vars <- common_vars[
    sapply(full_x[common_vars], function(x) dplyr::n_distinct(x) > 1)
  ]
  
  full_x <- full_x %>% dplyr::select(Target, dplyr::all_of(keep_vars))
  pred_x <- pred_x %>% dplyr::select(ID, dplyr::all_of(keep_vars))
  
  final_formula <- reformulate(keep_vars, response = "Target")
  
  fit <- glm(
    final_formula,
    data = full_x,
    family = binomial(),
    control = glm.control(maxit = 100)
  )
  
  p_pred <- predict(fit, newdata = pred_x, type = "response")
  p_pred <- pmin(pmax(p_pred, 1e-6), 1 - 1e-6)
  
  cat("\n", label, "- final vars:\n")
  print(all.vars(formula(fit))[-1])
  
  tibble(
    ID = pred_x$ID,
    Pred = p_pred
  )
}

# =========================================================
# 5. MEN - STEP AIC BOTH + COOK
# =========================================================
m_eval <- run_stepwise_glm(
  train_df = m_train,
  valid_df = m_valid,
  pred_df = m_pred,
  candidate_vars = men_candidates,
  label = "MEN stepAIC both"
)

# =========================================================
# 6. WOMEN - STEP AIC BOTH + COOK
# =========================================================
w_eval <- run_stepwise_glm(
  train_df = w_train,
  valid_df = w_valid,
  pred_df = w_pred,
  candidate_vars = women_candidates,
  label = "WOMEN stepAIC both"
)

# =========================================================
# 7. REFIT NA 2014-2025 I PREDYKCJA 2026
# =========================================================
m_sub <- refit_on_full_train(
  full_df = m_full,
  pred_df = m_pred,
  selected_vars = m_eval$selected_vars,
  label = "MEN full train 2014-2025"
)

w_sub <- refit_on_full_train(
  full_df = w_full,
  pred_df = w_pred,
  selected_vars = w_eval$selected_vars,
  label = "WOMEN full train 2014-2025"
)

submission <- bind_rows(m_sub, w_sub) %>%
  arrange(ID)

write_csv(submission, file.path(DIR_OUT, "submission_stepAIC_both.csv"))

# =========================================================
# 8. PODSUMOWANIE
# =========================================================
metrics_all <- bind_rows(m_eval$metrics, w_eval$metrics)
print(metrics_all)

cat("\nPlik submission zapisany do:\n")
cat(file.path(DIR_OUT, "submission_stepAIC_both.csv"), "\n")