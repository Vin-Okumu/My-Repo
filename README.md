# Climate Change Analysis
Climate change modeling is an essential tool for understanding and predicting the complex interactions among various climate indicators, including atmospheric carbon dioxide (CO₂) concentration, global temperature changes, sea level rise, and Arctic ice coverage. These variables are interconnected, with shifts in one often driving or amplifying changes in the others, necessitating comprehensive modeling approaches to assess long-term trends and future impacts.

Over the past century, anthropogenic activities—such as fossil fuel combustion, deforestation, and industrial emissions—have significantly increased atmospheric CO₂ levels. This rise in greenhouse gases enhances the greenhouse effect, contributing to a persistent increase in global temperatures. Climate models simulate these processes to quantify the effects of rising CO₂ on temperature patterns and predict potential future scenarios based on varying emission trajectories.

By leveraging climate models, this analysis aims to test six key hypotheses regarding the relationships between the four variables measured for this hypothetical dataset; CO₂ concentration, global average temperature, sea level rise, and Arctic ice area coverage. Understanding these interconnections through predictive modeling is crucial for informing climate mitigation and adaptation strategies, ensuring a data-driven approach to addressing climate change challenges.

```r
library (tidyverse)
df = read.csv("D:/Analytics/Climate_Change/Climate_Change_Indicators.csv") #reads in the data frame 
colnames(df) = c("Year","Global_Avg_Temp","CO2_Conc","Rise_in_Sea_Lvl","Arctic_Ice_Area") #renames the columns
str(df) #Returns the summary of the data frame
```
```r
anyNA(df) #Checks for missing values, helps ensure there are no missing values within data frame
head(df) #Returns a glimpse of the data frame
```
# Rearranging Data

```r
library(dplyr) 
df = df%>%
  arrange(Year) #Rearranges the data in the data frame by year
head(df)
tail(df)
```
# Data Summary

Since the original data contains several readings per variable per year, it was logical to assume that the records stood for readings measured in different parts of the globe at different times of the year. Therefore, averaging them out gives a proper perspective of the estimated readings at any given time of the year or part of the globe as a mean reading.

```r
df = df %>%
  group_by(Year) %>%
  summarise(across(everything(),\(x)mean(x, na.rm = TRUE))) #calculates the mean readings for each year respectively
head(df)
tail(df)
```
# Check for Normality

Before performing any analyses, perhaps it would be prudent to test the data that we have imported for normality. You may ask, "Is that any important?" Yes is is important for various reasons, especially since we intend to perform inferential analysis using the data set, and different statistical tests assume the data follows a normal distribution.

**Why test for normality**

Effect on Correlation and Regression – Since we intend to check the association between the variables, it's important to note tha normality affects how well correlation and regression models fit the data. Non-normal data may produce misleading correlation coefficients and biased regression estimates.

**Validity of Parametric Tests** – Many inferential tests, such as t-tests, ANOVA, and linear regression, assume that the data (or residuals) are normally distributed. Violating this assumption can lead to incorrect conclusions.

* **Accuracy of Confidence Intervals and p-values** – If the data are not normal, confidence intervals and p-values in parametric tests may be inaccurate, leading to potential Type I (false positive) or Type II (false negative) errors.

* **Normality Test using Shapiro-Wilk Test**

For this session, we are testing the data for normality using Shapiro-Wilk test, especially since we had already summarized the data to contain the mean readings for each year instead of the original repeated readings. Therefore, the resultant dataset contains records less than 5000.


```r
# Check normality using Shapiro-Wilk test
for (col in c("Global_Avg_Temp", "CO2_Conc", "Rise_in_Sea_Lvl", "Arctic_Ice_Area")) {
  cat("\nShapiro-Wilk test for", col, "\n")
  print(shapiro.test(df[[col]]))
}
```
With Shapiro-Wilk test, the hypothesis being tested is as follows:

* **Null hypothesis (H0):** The data is normally distributed

* **Alternative hypothesis (H1):** The data is not normally distributed

In cases where we fail to reject the null hypothesis (meaning the p_value is calculated to be greter than 0.05) the test statistic shows us how normal the data is. In such a case, values closer to 0 indicate the data is less normal and values closer to 1 indicate the data is more normally distributed.

From the Shapiro-Wilk test results above, it is evident that the all variables are normally distributed as the respective p_values are greater than 0.05. Additionally, with each statistic closer to 1, it shows that the data are more normally distributed as will be proven by the following histogram plots.

```r
library(ggplot2)
# Convert data to long format
df_long <- pivot_longer(df, cols = c(Global_Avg_Temp, CO2_Conc, Rise_in_Sea_Lvl, Arctic_Ice_Area), 
                        names_to = "Variable", values_to = "Value")

# Plot histograms using facets
ggplot(df_long, aes(x = Value)) +
  geom_histogram(aes(y = after_stat(density)), bins = 10, fill = "goldenrod4", alpha = 0.5) +
  geom_density(color = "grey0", linewidth = 0.5) +  # Density curve
  facet_wrap(~ Variable, scales = "free") +  # Separate histograms for each variable
  theme_minimal() +
  labs(title = "Histograms of Readings", x = "Value", y = "Density") +
  theme(
      strip.text = element_text(size = 14, face = "bold", color = "black"),
      panel.spacing = unit(1, "cm"),  #increase space between facets
      panel.grid.major = element_blank(), #removes major  gridlines
      panel.grid.minor = element_blank(),
      axis.title = element_text(color = "black", size = 18), #colors the axis titles bisque
      plot.title = element_text(color = "black", face = "bold", hjust = 0.5, size = 20)) #removes minor gridlines
      options(repr.plot.width = 12, repr.plot.height = 8) #sets panel width and height

```
From the statistics above and the accompanying histograms, it is evident that all variable readings follow a normal distribution. This allows us to proceed with the next step of the analysis, estimating the correlation between the variables to see how they relate to the underlying hypothesis.

# Testing Correlation

With four variables measured in this dataset, and the underlying need for analysis remaining to establish the association between these variables, it is important to embark on testing the correlation between each variable.

**Testing Correlation between CO₂ Concentration and Global Average Temperature**

The first set of variables to be tested were atmospheric carbon dioxide (CO₂) concentration and global average temperature. The common assumption has always been that a rise in atmospheric carbon dioxide (CO₂) concentration leads to an increase in global average temperature, which is the resultant cause of global warming.
```r
ggplot(df,aes(x = CO2_Conc, y = Global_Avg_Temp)) +
  geom_point(pch = 15, color = "firebrick3", size = 1.5) +
  ggtitle("CO2 Concentration vs Global Average\nTemperature") +
  scale_x_continuous(expand = c(0.01,0.01)) +
  geom_smooth(method = "lm", se = FALSE, color = "darkblue", linewidth = 0.4) +
  theme_minimal(base_size = 15) +
  theme(
    plot.background = element_rect(fill = "grey85", color = NA), #greys out the background
    panel.background = element_rect(fill = "gray99", color = NA), #colors the panel white
    panel.grid.major = element_blank(), #removes major  gridlines
    panel.grid.minor = element_blank(), #removes minor gridlines
    axis.text = element_text(color = "black"), #colors the axis labels lavender
    axis.title = element_text(color = "gray25", face = "bold"), #colors the axis titles bisque
    plot.title = element_text(color = "grey10", face = "bold", hjust = 0.5, size = 20) #Colors the title gold, bold it and centers it across the canvas
  )
  
```
However, from the above plot, the underlying impication is that global average temperature is negatively correlated with CO₂ concentration. The above plot insinuates that as the CO₂ concentration increases, the global average temperature decreases. Therefore, to ascertain the given correlation, it would be logical to perform a correlation test. 

```r
#Performing pearson correlation test
Corr_1 = cor.test(df$Global_Avg_Temp,df$CO2_Conc, method = "pearson")
print(Corr_1)

#Extracting P-value
p_value = Corr_1$p.value

#Hypothesis
if (p_value < 0.05) {
  print("There is a significant correlation between Global average temperature and CO2 concentration.")
} else {
  print("There is no significant correlation between Global average temperature and CO2 concentration.")
}
```
**Results from Pearson Correlation Test**

For this analysis we tested correlation using Pearson Correlation test, which returns the results that there is no significant correlation between Global average temperature and CO₂ concentration. This result implies that whatever correlation was determined from the plot may have happened by chance. This finding would make sense given the dataset was simulated. 
**Testing Correlation between Global Average Temperature and Rise in Sea level**

The second set of variables to be tested were global average temperature and rise in sea level. Similar to the assumption asserted when looking at CO₂ concentration and Global average temperature, the assumption here is that a rise in global average temperature leads to a rise in sea level, which is the result of global warming.

```r
ggplot(df,aes(x = Global_Avg_Temp, y = Rise_in_Sea_Lvl)) +
  geom_point(pch = 8, color = "azure", size = 2) +
  ggtitle("Global Average Temperature vs Rise in Sea\nLevel") +
  scale_x_continuous(expand = c(0.001,0.001)) +
  geom_smooth(method = "lm", se = FALSE, color = "ivory", linewidth = 0.4) +
  theme_minimal(base_size = 15) +
  theme(
    plot.background = element_rect(fill = "grey5", color = NA), #greys out the background
    panel.background = element_rect(fill = "gray8", color = NA), #colors the panel white
    panel.grid.major = element_blank(), #removes major  gridlines
    panel.grid.minor = element_blank(), #removes minor gridlines
    axis.text = element_text(color = "lavender"), #colors the axis labels lavender
    axis.title = element_text(color = "bisque", face = "bold"), #colors the axis titles bisque
    plot.title = element_text(color = "ivory", face = "bold", hjust = 0.5, size = 20) #Colors the title gold, bold it and centers it across the canvas
  )
```
From the above plot, the common assumption is relatively supported, with the plot sublty implying that global average temperature is weakly positively correlated with rise in sea level. The above plot, therefore, insinuates that as the global average temperature increases, the sea level rises. However, to ascertain the given correlation, it would be logical, again, to perform a correlation test.  

```r
Corr_2 = cor.test(df$Global_Avg_Temp,df$Rise_in_Sea_Lvl, method = "pearson")
print(Corr_2)

#Extracting P-value
p_value = Corr_2$p.value

#Hypothesis

if (p_value < 0.05) {
  print("There is a significant correlation between Global average temperature and Rise in sea level.")
} else {
  print("There is no significant correlation between Global average temperature and Rise in sea level.")
}
```
**Results from Pearson Correlation Test**

Similar to the previous test, we performed a Pearson Correlation test, which returns the results that there is no significant correlation between Global average temperature and rise in sea level. This result implies that whatever correlation was determined from the above plot may have happened by chance. Again, this finding would make sense given the datase was simulated.

**Testing Correlation between Global Average Temperature and Arctic Ice Area**

The third set of variables to be tested were global average temperature and arctic ice area. Similar to the previously asserted assumptions, the assumption here is that a rise in global average temperature leads to a decrease in arctic ice area as a result of ice melting.

```r
ggplot(df,aes(x = Global_Avg_Temp, y = Arctic_Ice_Area)) +
  geom_point(pch = 8, color = "darkblue", size = 2) +
  ggtitle("Global Average Temperature vs Arctic Ice Area") +
  scale_x_continuous(expand = c(0.001,0.001)) +
  geom_smooth(method = "lm", se = FALSE, color = "firebrick4", linewidth = 0.4) +
  theme_minimal(base_size = 15) +
  theme(
    plot.background = element_rect(fill = "grey45", color = NA), #greys out the background
    panel.background = element_rect(fill = "grey95", color = NA), #colors the panel white
    panel.grid.major = element_blank(), #removes major  gridlines
    panel.grid.minor = element_blank(), #removes minor gridlines
    axis.text = element_text(color = "lavender"), #colors the axis labels lavender
    axis.title = element_text(color = "bisque", face = "bold"), #colors the axis titles bisque
    plot.title = element_text(color = "darkorange2", face = "bold", hjust = 0.5, size = 20) #Colors the title gold, bold it and centers it across the canvas
  )

```
However, from the above plot, the assumption is not suported as the plot sublty implies that global average temperature is weakly positively correlated with arctic ice area. The above plot, therefore, insinuates that as the global average temperature increases, the arctic ice area increases. Therefore, to ascertain the given correlation, it would be logical, again, to perform a correlation test. 

```r
Corr_3 = cor.test(df$Global_Avg_Temp,df$Arctic_Ice_Area, method = "pearson")
print(Corr_3)

#Extracting P-value
p_value = Corr_3$p.value

#Hypothesis

if (p_value < 0.05) {
  print("There is a significant correlation between Global average temperature and Arctic ice area.")
} else {
  print("There is no significant correlation between Global average temperature and Arctic ice area.")
}

```
**Results from Pearson Correlation Test**

Similar to the previous tests, we performed a Pearson Correlation test, which returns the results that there is no significant correlation between Global average temperature and arctic ice area. This result implies that whatever correlation was determined from the above plot may have happened by chance. Again, this finding would make sense given the data was simulated. 

**Testing Correlation between CO₂ Concentration and Arctic Ice Area**

The fourth set of variables whose correlation was tested was CO₂ concentration and arctic ice area. Similar to the previously asserted assumptions, the assumption here is that an increase in CO₂ concentration leads to a decrease in arctic ice area as a result of ice melting from global warming.

```r
ggplot(df,aes(x = CO2_Conc, y = Arctic_Ice_Area)) +
  geom_point(pch = 17, color = "black", size = 2) +
  ggtitle("CO2 Concentration vs Arctic Ice Area") +
  scale_x_continuous(expand = c(0.001,0.001)) +
  geom_smooth(method = "lm", se = FALSE, color = "firebrick4", linewidth = 0.4) +
  theme_minimal(base_size = 15) +
  theme(
    plot.background = element_rect(fill = "grey4", color = NA), #greys out the background
    panel.background = element_rect(fill = "aliceblue", color = NA), #colors the panel white
    panel.grid.major = element_blank(), #removes major  gridlines
    panel.grid.minor = element_blank(), #removes minor gridlines
    axis.text = element_text(color = "lavender"), #colors the axis labels lavender
    axis.title = element_text(color = "bisque", face = "bold"), #colors the axis titles bisque
    plot.title = element_text(color = "ivory", face = "bold", hjust = 0.5, size = 20) #Colors the title gold, bold it and centers it across the canvas
  )

```
From the above plot, the common assumption is not supportted, with the plot sublty implying that CO2 concentration is weakly positively correlated with arctic ice area. The above plot, therefore, insinuates that as the CO2 concentration increases, the arctic ice area increases slightly. However, again, to ascertain the given correlation, it would be logical to perform a correlation test. 



```r
Corr_4 = cor.test(df$CO2_Conc,df$Arctic_Ice_Area, method = "pearson")
print(Corr_4)

#Extracting P-value
p_value = Corr_4$p.value

#Hypothesis

if (p_value < 0.05) {
  print("There is a significant correlation between CO2 concentration and Arctic ice area.")
} else {
  print("There is no significant correlation between CO2 concentration and Arctic ice area.")
}
```
**Results from Pearson Correlation Test**

Similar to the previous tests, results from the Pearson Correlation test implied there is no significant correlation between CO2 concentration and arctic ice area. This result insinuate that whatever correlation was determined from the above plot may have happened by chance. Again, this finding would make sense given the data was simulated. 

**Testing Correlation between CO₂ Concentration and Rise in Sea Level**

The fifth set of variables to be tested were CO₂ concentration and rise in sea level. Here, the underlying assumption is always that an increase in CO₂ concentration leads to a rise in sea level as a result of ice melting from global warming.

```r
ggplot(df,aes(x = CO2_Conc, y = Rise_in_Sea_Lvl)) +
  geom_point(pch = 8, color = "darkgreen", size = 2) +
  ggtitle("CO2 Concentration vs Rise in Sea Level") +
  scale_x_continuous(expand = c(0,0)) +
  geom_smooth(method = "lm", se = FALSE, color = "firebrick4", linewidth = 0.4) +
  theme_minimal(base_size = 15) +
  theme(
    plot.background = element_rect(fill = "white", color = NA), #greys out the background
    panel.background = element_rect(fill = "grey98", color = NA), #colors the panel white
    panel.grid.major = element_blank(), #removes major  gridlines
    panel.grid.minor = element_blank(), #removes minor gridlines
    axis.text = element_text(color = "darkgreen"), #colors the axis labels lavender
    axis.title = element_text(color = "darkgreen", face = "bold"), #colors the axis titles bisque
    plot.title = element_text(color = "limegreen", face = "bold", hjust = 0.5, size = 20) #Colors the title gold, bold it and centers it across the canvas
  )
```
From the above plot, the common assumption is supportted, with the plot implying that CO2 concentration is positively correlated with rise in sea level. The above plot, therefore, insinuates that as the CO2 concentration increases, the sea level rises. However, again, to ascertain the given correlation, it would be logical to perform a correlation test. 

```r
Corr_5 = cor.test(df$CO2_Conc,df$Rise_in_Sea_Lvl, method = "pearson")
print(Corr_5)

#Extracting P-value
p_value = Corr_5$p.value

#Hypothesis

if (p_value < 0.05) {
  print("There is a significant correlation between CO2 concentration and Rise in sea level.")
} else {
  print("There is no significant correlation between CO2 concentration and Rise in sea level.")
}
```
**Results from Pearson Correlation Test**

Like the previous tests, results from the Pearson Correlation test implied there is no significant correlation between CO₂ concentration and rise in sea level. This result insinuate that whatever correlation was determined from the above plot may have happened by chance. Again, this finding would make sense given the data was simulated. 

**Testing Correlation between Arctic Ice Area and Rise in Sea Level**

The final set of variables to be tested were arctic ice area and rise in sea level. Here, the underlying assumption is always that a decrease in arctic ice area leads to a rise in sea level as a result of ice melting.

```r
ggplot(df,aes(x = Arctic_Ice_Area, y = Rise_in_Sea_Lvl)) +
  geom_point(pch = 18, color = "grey1", size = 3.5) +
  ggtitle("Arctic Ice Area vs Rise in Sea Level") +
  scale_x_continuous(expand = c(0.001,0.001)) +
  geom_smooth(method = "lm", se = FALSE, color = "firebrick4", linewidth = 0.4) +
  theme_minimal(base_size = 15) +
  theme(
    plot.background = element_rect(fill = "grey0", color = NA), #greys out the background
    panel.background = element_rect(fill = "grey98", color = NA), #colors the panel white
    panel.grid.major = element_blank(), #removes major  gridlines
    panel.grid.minor = element_blank(), #removes minor gridlines
    axis.text = element_text(color = "lavender"), #colors the axis labels lavender
    axis.title = element_text(color = "bisque", face = "bold"), #colors the axis titles bisque
    plot.title = element_text(color = "cornsilk", face = "bold", hjust = 0.5, size = 20) #Colors the title gold, bold it and centers it across the canvas
  )
```
From the above plot, the common assumption is not supportted since the plot implies that arctic ice area is positively correlated with rise in sea level. The above plot, therefore, insinuates that as the arctic ice area increased the sea level rises. However, again, to ascertain the given correlation, it would be logical to perform a correlation test. 

```r
Corr_6 = cor.test(df$Arctic_Ice_Area,df$Rise_in_Sea_Lvl, method = "pearson")
print(Corr_6)

#Extracting P-value
p_value = Corr_6$p.value

#Hypothesis

if (p_value < 0.05) {
  print("There is a significant correlation between Arctic ice area and Rise in sea level.")
} else {
  print("There is no significant correlation between Arctic ice area and Rise in sea level.")
}
```
**Results from Pearson Correlation Test**

Like the previous tests, results from the Pearson Correlation test implied there is no significant correlation between arctic ice area and rise in sea level. The result insinuate that whatever correlation was determined from the above plot may have happened by chance. Again, this finding would make sense given the data was simulated. 

# Conclusion

From the analysis above, it would be accurate to deduce that, according to the data set provided CO₂ concentration is not correlated to Global average temperature. Neither is Global average temperature correlated to Sea level or Arctic area. Therefore, with the statistics implying that the previously determined correlations  happened by chance, it would not be helpful to continue further with data modeling to predict future climate change outcomes knowing the variables do not directly associate. Therefore, to move forward with simulating the implications of these variables on overall climate change, it would be prudent to utilize real world data from platforms like NOAA.

```{r}

```

