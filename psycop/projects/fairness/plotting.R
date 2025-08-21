setwd("E:/sara/data/psycop-common")

library(ggplot2)

df = read.csv("metrics_1000_auroc.csv")

df = df[,-1]
# df = df %>% filter(variable != "Percentage")

ggplot(df, aes(x=variable, y=value, ymin=lower, ymax=upper, fill=Model, alpha=Sex)) +
  geom_bar(stat="identity", position="dodge") +
  geom_errorbar(position="dodge") + 
  scale_alpha_manual(values = c("Female" = 1, "Male" = 0.3)) +
  scale_fill_manual(values = c("#C96868", "#A2678A")) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 10, hjust = 1, vjust = 0.5)) +
  labs(title="Fairness metrics", x="Metric", y="Value")


df_wide <- df %>%  pivot_wider(names_from = Sex, values_from = c(value, lower, upper))

df_ratio <- df_wide %>%
  mutate(
    mean_ratio = value_Male / value_Female,
    lower_ratio = lower_Male / lower_Female,
    upper_ratio = upper_Male / upper_Female
  )


df_ratio1 <- df_ratio %>%
  mutate(
    ratio_diff = mean_ratio - 1,
    lower_ratio = lower_ratio - 1,
    upper_ratio = upper_ratio - 1)

test = c(" ", " ", "*", " ", "*", " ", "*", " ", "*", " ", "*", " ", "*", " ")

df_ratio1$significance = test

ggplot(df_ratio1, aes(x=variable, y=ratio_diff, ymin=lower_ratio, ymax=upper_ratio, fill=Model)) +
  geom_col( position="dodge") +
  # geom_errorbar(position="dodge") +
  scale_fill_manual(values = c("#C96868", "#A2678A")) +
  geom_text(aes(label = significance), nudge_x = 0.2) +
  theme_bw() +
  labs(title="Fairness metrics", x=" ", y="Value") +
  ylim(0.5, 1.5) +
  scale_y_continuous(
    labels = function(x) round(x + 1, 2),  # Convert back to ratio scale for display
    name = "Relative metric ratio (male to female)"
  ) +
  coord_flip()

