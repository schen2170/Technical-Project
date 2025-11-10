#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: Predict_House_Valued.ipynb
Conversion Date: 2025-11-10T19:15:36.639Z
"""

# **Import all the necessary modules.**


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# **Since the CSV file is 61.77 MB and is too big for local file upload or Github upload, we must upload it through Google Drive files.**


link = "https://github.com/SamuelChen2170/sam/releases/download/v1.0/blkjckhands.csv"
df = pd.read_csv(link)

# # **Preliminary Examination of Data**


df.head()

df.info()

df['winloss'].value_counts()

df['blkjck'].value_counts()

# Based on a preliminary examination of the given Blackjack dataset, I identified an intriguing question to explore. In Blackjack, a fundamental assumption is that certain hands, especially those approaching 21, give the player an advantage. The common perception is that **attaining a high-value hand is worth the risk, as it increases the chances of winning**. However, this assumption may warrant closer scrutiny. Does a high hand value always correlate with winning, or are there exceptions? Does the player’s probability of success vary depending on the dealer’s hand or initial card? Furthermore, are certain hand values, or even specific card combinations, more advantageous than others? I aim to delve deeper into the **effectiveness of different hands and strategies across various game scenarios** to understand decision-making in this context.


# # **Research Question: With the same rule as casino Blackjack, what are the odds of the player winning given various combinations of cards?**
# 
# ##### To answer this question, I will examine the following sub-questions in my analysis.
# - What are the odds of winning?
# - How does player's initial hand influence the likelihood of the player winning?
# - How does player's final hand influence the likelihood of the player winning?
# - How does dealer's various hand influence the likelihood of the player winning?
# 
# 


# #**1) What are the odds of winning?**


# # **How Frequent Does the Player Win Against the House?**


player_win_rate = (df['winloss'] == 'Win').mean() * 100
print(f"Player Win Percentage: {player_win_rate}%")

# # **How Frequent Does the House Win Against the Player?**


house_win_rate = (df['winloss'] == 'Loss').mean() * 100
print(f"House Win Percentage: {house_win_rate}%")

# # **How Frequent does Both the Player and Dealer Push?**


tie_rate = (df['winloss'] == 'Push').mean() * 100
print(f"Push Percentage: {tie_rate}%")

# # **Table With the Odds of Winning**


player_win_rate = (df['winloss'] == 'Win').mean() * 100
house_win_rate = (df['winloss'] == 'Loss').mean() * 100
push_rate = (df['winloss'] == 'Push').mean() * 100

summary_table = pd.DataFrame({
    '': ['Player Win Percentage', 'House Win Percentage', 'Push Percentage'],
    'Percentage (%)': [player_win_rate, house_win_rate, push_rate]
})

summary_table

# Historically, the odds for Blackjack: **Player Win Percentage (42.22%)**, **House Win Percentage (49.10%)**, and **Push Percentage (8.48%)**.
# 
# The table presents three key percentages from this sample of **900,000** games: **Player Win Percentage (42.88%)**, **House Win Percentage (47.76%)**, and **Push Percentage (9.36%)**. These values give us an overview of game outcomes and hint at underlying game dynamics in Blackjack.
# 
# - House Advantage: The House Win Percentage is notably higher than the Player Win Percentage **(47.76% vs. 42.88%)**, indicating an inherent advantage for the house. This is expected in Blackjack, where casinos often have rules that slightly favor the dealer, such as the dealer winning ties or mandatory dealer actions (e.g., hitting on soft 17). The nearly **5%** higher win rate for the house reinforces this structural advantage and suggests that, over a large number of games, players will statistically lose more often than they win.
# - Push Percentage: The Push Percentage **(9.36%)** represents cases where both the player and dealer end with the same hand value, resulting in no winner. Although push outcomes are less frequent than wins or losses, the **9.36%** rate indicates that nearly **1 in 10** games ends in a tie. This outcome provides a slight buffer for players, as pushes don’t result in a loss.
# - Implications for Player Strategy: With the player win rate below the house win rate, it may be beneficial for players to adopt strategies that mitigate the house's edge. For instance, understanding when to stand, hit, or double down based on the dealer’s visible card could help players maximize their win chances, as well as minimize losses. Additionally, the push rate suggests that there’s a chance of reaching a tied outcome, which could influence a player’s decision-making, particularly when close to 21.


# # **2) How does player's various hand influence the likelihood of the player winning?**


# # **Frequency of Initial Player Hand**


# Add a new column for the sum of the player's first two cards
df['sum_card1_card2'] = df['card1'] + df['card2']

# Set up the plot figure and create a histogram to show the frequency of each initial hand sum
plt.figure(figsize=(10, 6))
plt.hist(df['sum_card1_card2'], bins=range(1, 23), edgecolor='black', align='left')
plt.title('Histogram of Frequency of Initial Player Hand')
plt.xlabel('Sum of Card 1 and Card 2')
plt.ylabel('Frequency')
plt.xticks(range(0, 22))

# Display the plot
plt.show()

# This histogram displays the frequency of initial player hands in Blackjack, where the x-axis represents the sum of the player’s first two cards, and the y-axis represents the frequency of each hand sum.
# 
# ### Analysis:
# 
# - **Most Common Hand Sums**: The sums around 12 to 16 are the most frequently occurring initial hands, with a particularly high frequency around 12 and 13. This is significant because these values place the player in a challenging position, often referred to as the “danger zone,” where hitting risks a bust but standing provides a weak hand against the dealer.
# 
# - **High Frequency of 20**: There is a notable peak at 20, which suggests that a two-card sum of 20 (likely consisting of face cards or 10-value cards) is relatively common. A starting hand of 20 is highly favorable, as it puts the player in a strong position without requiring further action. This finding supports the general strategy of standing on 20, as it’s a near-optimal hand.
# 
# - **Frequency Drop at 21**: Interestingly, the frequency drops from 20 to 21. This is likely because 21 as an initial hand can only be achieved by Blackjack---making it less frequent. Achieving 21 from the first two cards offers the best odds for an automatic win, or at the very least push.
# 
# - **Lower Frequency of Low Sums**: The lower sums (1 to 6) have minimal frequencies, indicating that initial hands with very low values are uncommon in this dataset. This is logical since the starting hand typically involves higher cards that tend to sum above 6. Low starting values are generally less advantageous, often requiring multiple hits to reach a competitive hand---which is riskier.
# 
# ### Implications for Strategy:
# 
# This distribution of initial hand sums informs player strategy by highlighting the common hand values and potential risk points. For instance, the high frequency of “danger zone” hands (12-16) suggests that players frequently encounter tough decisions on whether to hit or stand. Meanwhile, the higher frequency of 20 supports the strategy of standing on strong initial hands to avoid the risk of busting.


# ## **Stacked Bar Graph of Win/Loss/Push by Inital Player Hand**


# Create a DataFrame to calculate win, loss, and push percentages by player hand sum
df['sum_first_two_cards'] = df['card1'] + df['card2']
win_loss_data = df.groupby('sum_first_two_cards')['winloss'].value_counts(normalize=True).unstack() * 100

# Define the color map for each outcome
color_map = {'Win': 'green', 'Loss': 'red', 'Push': 'grey'}

# Filter outcomes to only those available in the data
available_outcomes = [outcome for outcome in color_map if outcome in win_loss_data.columns]
colors = [color_map[outcome] for outcome in available_outcomes]

# Plotting the stacked bar chart
win_loss_data[available_outcomes].plot(kind='bar', stacked=True, figsize=(12, 6), color=colors)
plt.title('Win/Lose/Push Breakdown by Initial Player Hand')
plt.xlabel('Player Hand Sum')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=0)
plt.legend(title='Outcome')
plt.show()

# # **Win Rate by Player’s Initial Hand**


# Calculate the sum of the player's first two cards and store it in a new column
df['sum_first_two_cards'] = df['card1'] + df['card2']

# Convert win/loss outcomes to binary (1 for Win, 0 for Loss) in a new column
df['winloss_binary'] = df['winloss'].apply(lambda x: 1 if x == 'Win' else 0)

# Group by the sum of the first two cards and calculate the win rate as a percentage
win_rate_by_sum = df.groupby('sum_first_two_cards')['winloss_binary'].mean() * 100

# Set up the plot figure and create a line plot to display win rate by initial hand sum
plt.figure(figsize=(10, 6))
plt.plot(win_rate_by_sum.index, win_rate_by_sum.values, marker='o', color='Green')
plt.title('Win Rate by Player’s Initial Hand')
plt.xlabel('Sum of First Two Cards')
plt.ylabel('Win Rate (%)')
plt.xticks(range(2, 22))
plt.grid(True)

# Display the plot
plt.show()

# # **Lose Rate by Player’s Initial Hand**


# Calculate the sum of the player's first two cards and store it in a new column
df['sum_first_two_cards'] = df['card1'] + df['card2']

# Convert win/loss outcomes to binary (1 for Loss, 0 for Win) in a new column
df['winloss_binary'] = df['winloss'].apply(lambda x: 1 if x == 'Loss' else 0)

# Group by the sum of the first two cards and calculate the loss rate as a percentage
win_rate_by_sum = df.groupby('sum_first_two_cards')['winloss_binary'].mean() * 100

# Set up the plot figure and create a line plot to display loss rate by initial hand sum
plt.figure(figsize=(10, 6))
plt.plot(win_rate_by_sum.index, win_rate_by_sum.values, marker='o', color='Red')
plt.title('Lose Rate by Player’s Initial Hand')
plt.xlabel('Sum of First Two Cards')
plt.ylabel('Lose Rate (%)')
plt.xticks(range(2, 22))
plt.grid(True)

# Display the plot
plt.show()

# # **Push Rate by Player’s Initial Hand**


# Calculate the sum of the player's first two cards and store it in a new column
df['sum_first_two_cards'] = df['card1'] + df['card2']

# Convert win/loss outcomes to binary (1 for Push, 0 for other outcomes) in a new column
df['winloss_binary'] = df['winloss'].apply(lambda x: 1 if x == 'Push' else 0)

# Group by the sum of the first two cards and calculate the push rate as a percentage
win_rate_by_sum = df.groupby('sum_first_two_cards')['winloss_binary'].mean() * 100

# Set up the plot figure and create a line plot to display push rate by initial hand sum
plt.figure(figsize=(10, 6))
plt.plot(win_rate_by_sum.index, win_rate_by_sum.values, marker='o', color='Grey')
plt.title('Push Rate by Player’s Initial Hand')
plt.xlabel('Sum of First Two Cards')
plt.ylabel('Push Rate (%)')
plt.xticks(range(2, 22))
plt.grid(True)

# Display the plot
plt.show()

df['sum_first_two_cards'] = df['card1'] + df['card2']

# Calculate Win Rate by Initial Two-Card Sum
df['winloss_binary'] = df['winloss'].apply(lambda x: 1 if x == 'Win' else 0)
win_rate_by_sum = df.groupby('sum_first_two_cards')['winloss_binary'].mean() * 100

# Calculate Push Rate by Initial Two-Card Sum
df['winloss_binary'] = df['winloss'].apply(lambda x: 1 if x == 'Push' else 0)
push_rate_by_sum = df.groupby('sum_first_two_cards')['winloss_binary'].mean() * 100

# Calculate Lose Rate by Initial Two-Card Sum
df['winloss_binary'] = df['winloss'].apply(lambda x: 1 if x == 'Loss' else 0)
lose_rate_by_sum = df.groupby('sum_first_two_cards')['winloss_binary'].mean() * 100

# Plot all three rates on the same graph
plt.figure(figsize=(10, 6))
plt.plot(win_rate_by_sum.index, win_rate_by_sum.values, marker='o', color='green', label='Win Rate')
plt.plot(push_rate_by_sum.index, push_rate_by_sum.values, marker='o', color='grey', label='Push Rate')
plt.plot(lose_rate_by_sum.index, lose_rate_by_sum.values, marker='o', color='red', label='Lose Rate')
plt.title('Win, Push, and Lose Rates by Player’s Initial Hand')
plt.xlabel('Sum of First Two Cards')
plt.ylabel('Rate (%)')
plt.xticks(range(2, 22))
plt.grid(True)
plt.legend()
plt.show()

# This chart provides insights into the relationship between initial hand sums and outcome probabilities (win, push, and lose rates) in Blackjack. Notably, the points where the lines cross reveal pivotal shifts in player advantage and risk.
# 
# ### Analysis:
# 
# - **Win and Lose Rate Cross-Over (Around 18)**: The win and lose rates intersect at an initial hand sum of approximately **18**. This crossing point marks a transition where higher initial hand sums start favoring the player, as win rates begin to exceed lose rates. This point aligns with the typical Blackjack strategy of standing on hands of **18** or higher, as the the dealer must stand at **17** or greater.
# 
# - **Stable Push Rate**: The push rate remains relatively low and stable across most initial hands, with only a slight increase around hands **17-20**. This suggests that ties are generally uncommon, except when both the player and dealer hold high-value hands close to **21**.
# 
# - **Divergence at High Hands (20 and 21)**: As initial hand sums reach **20** and **21**, the win rate spikes sharply, while the lose rate declines to nearly **zero**. This divergence highlights the substantial advantage these high sums provide, virtually guaranteeing a win with an initial **21**. This supports the notion that achieving a hand of **20 or 21** offers a significant edge, creating a clear strategic advantage.
# 
# 
# Focusing on the points where win and lose rates cross or diverge provides deeper insight into optimal decision points in Blackjack. The intersection at **18** emphasizes the shift where standing becomes more favorable, while the divergence at **20-21** reinforces the value of high hands for maximizing winning outcomes. This analysis confirms strategic thresholds that can guide player decisions in the game.


# # **Linear Regression: Win Rate by Initial Player Card**


# Convert win/loss outcomes to binary values where 'Win' is 1, others are 0
df['winloss_binary'] = df['winloss'].apply(lambda x: 1 if x == 'Win' else 0)

# Calculate the sum of the player's first two cards
df['sum_first_two_cards'] = df['card1'] + df['card2']

# Group by initial two-card sum to calculate the average win rate for each hand sum
win_rate_by_initial_hand = df.groupby('sum_first_two_cards')['winloss_binary'].mean() * 100

# Prepare the data for numpy-based linear regression
X = win_rate_by_initial_hand.index.values  # Initial player hand sum (first two cards)
y = win_rate_by_initial_hand.values        # Win rate

# Perform linear regression using numpy
X_mean = np.mean(X)
y_mean = np.mean(y)
numerator = np.sum((X - X_mean) * (y - y_mean))
denominator = np.sum((X - X_mean) ** 2)
slope = numerator / denominator
intercept = y_mean - (slope * X_mean)

# Calculate the predicted values for plotting
predictions = slope * X + intercept

# Print the slope and intercept of the regression line
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")

# Plot the regression line with the data
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Win Rate Data')
plt.plot(X, predictions, color='red', label='Regression Line')
plt.xlabel('Sum of Initial Two Cards')
plt.ylabel('Player Win Rate (%)')
plt.title('Linear Regression: Win Rate by Initial Player Card')
plt.xticks(np.arange(min(X), max(X) + 1, 1))  # Set x-axis increments to 1
plt.legend()
plt.grid(True)
plt.show()

# This linear regression plot shows the relationship between the **sum of the initial two cards** and the **player’s win rate** in Blackjack. The blue dots represent actual win rate data points for each initial sum, and the red line indicates the linear regression line fitted to these data points, with a slope of approximately **1.50** and an intercept of about **23.94**.
# 
# ### Analysis:
# 
# - **Positive Correlation**: The positive slope **(1.50)** suggests a generally increasing trend in win rate as the initial hand sum rises. This indicates that higher initial sums are associated with a greater likelihood of winning, aligning with the intuitive understanding that stronger starting hands offer better chances in Blackjack.
# 
# - **Intercept Interpretation**: The intercept **(23.94)** represents the estimated win rate when the initial hand sum is **zero** **(hypothetically)**, which doesn’t have a practical implication here but helps anchor the regression model. It essentially means that even at lower hand sums, players have a baseline win probability influenced by factors outside just the initial hand sum.
# 
# - **Variation and Fit**: Although the regression line shows an overall upward trend, there’s noticeable variation in win rates for specific sums, especially between sums of **8 and 12**. For instance, certain sums like **10 and 11** show higher-than-expected win rates compared to nearby values, suggesting that specific hands may yield advantages or disadvantages that aren’t fully captured by a simple linear model.
# 
# - **Limitations of the Linear Model**: The win rate does not increase linearly across all hand sums. The regression line provides an approximation, but it may not fully reflect the complexity of Blackjack strategy. For example, win rates for sums above **17** plateau or vary less consistently, suggesting diminishing returns for higher initial sums.
# 
# ### Implications for Strategy:
# 
# This linear model gives a broad overview of how win rates improve with stronger initial hands. However, the variability seen in the actual data points indicates that other factors, such as dealer behavior and specific hand compositions, likely play a role in influencing outcomes. While higher initial sums generally correlate with higher win rates, players might benefit from strategies tailored to specific hands rather than relying solely on an upward trend.


# # **Linear Regression:  Bust Rate by Initial Player Card**


# Convert bust outcomes to binary values where 'Bust' is 1, others are 0
df['bust_binary'] = df['plybustbeat'].apply(lambda x: 1 if x == 'Bust' else 0)

# Calculate the sum of the player's first two cards
df['sum_first_two_cards'] = df['card1'] + df['card2']

# Group by initial two-card sum to calculate the average bust rate for each hand sum
bust_rate_by_initial_hand = df.groupby('sum_first_two_cards')['bust_binary'].mean() * 100

# Prepare the data for numpy-based linear regression
X = bust_rate_by_initial_hand.index.values  # Initial player hand sum (first two cards)
y = bust_rate_by_initial_hand.values        # Bust rate

# Perform linear regression using numpy
X_mean = np.mean(X)
y_mean = np.mean(y)
numerator = np.sum((X - X_mean) * (y - y_mean))
denominator = np.sum((X - X_mean) ** 2)
slope = numerator / denominator
intercept = y_mean - (slope * X_mean)

# Calculate the predicted values for plotting
predictions = slope * X + intercept

# Print the slope and intercept of the regression line
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")

# Plot the regression line with the data
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Bust Rate Data')
plt.plot(X, predictions, color='red', label='Regression Line')
plt.xlabel('Sum of Initial Two Cards')
plt.ylabel('Player Bust Rate (%)')
plt.title('Linear Regression:  Bust Rate by Initial Player Card')
plt.xticks(np.arange(min(X), max(X) + 1, 1))  # Set x-axis increments to 1
plt.legend()
plt.grid(True)
plt.show()

# This linear regression plot explores the relationship between the **sum of the initial two cards** and the **player's bust rate** in Blackjack. The blue points represent actual bust rate data for each initial hand sum, and the red line represents the regression line with a slope of **-1.35** and an intercept of **34.84**.
# 
# ### Analysis:
# 
# - **Negative Correlation**: The negative slope (**-1.35**) indicates an inverse relationship between the initial hand sum and the bust rate. As the sum of the initial two cards increases, the likelihood of the player busting decreases. This makes sense, as players with higher initial hands, such as **17 or above**, are more likely to stand rather than risk busting by hitting.
# 
# - **Intercept Interpretation**: The intercept (**34.84**) suggests that at an extremely low initial sum (hypothetically close to zero), the bust rate would be around **34.84%**.
# 
# - **Pattern of Bust Rates**: The regression line shows a clear decline in bust rate as initial hand values increase. For example, initial sums between **2 and 7 **have relatively high bust rates (around **30%**), as these hands typically require multiple hits to reach a competitive value, thereby increasing the risk of busting. In contrast, as initial hand values approach **17** or higher, the bust rate declines sharply, reaching close to **0%** for hands of **17** and above. This aligns with Blackjack strategy, where players often stand on high hands to avoid busting.
# 
# - **Deviation from the Regression Line**: Although the regression line captures the general downward trend, the actual data points show some deviation, especially for lower sums. For instance, bust rates vary slightly more for initial sums around **6 to 8**, which suggests that players with mid-range initial hands may sometimes employ different strategies, affecting the overall bust probability.
# 
# ### Implications for Strategy:
# 
# The inverse relationship between initial hand sums and bust rates depicts the risk associated with low starting hands. Players with low initial sums face higher bust probabilities due to the need for multiple hits to reach a competitive total. Meanwhile, higher initial sums, particularly **17** and above, provide a much safer position, supporting the strategy of standing on strong hands to minimize the chance of busting.


# # **3) How does player's final hand influence the likelihood of the player winning?**


# # **Frequency of Final Player Hand**


# Set up the figure size for the histogram
plt.figure(figsize=(10, 6))

# Plot a histogram for the final player hand values with bins from 1 to 24
plt.hist(df['sumofcards'], bins=range(1, 25), edgecolor='black', align='left')
plt.title('Histogram of Frequency of Final Player Hand')
plt.xlabel('Final Player Hand')
plt.ylabel('Frequency')
plt.xticks(range(1, 25))

# Display the plot
plt.show()

# This histogram displays the **frequency distribution** of **final player hand values** in Blackjack, where the x-axis represents final hand values and the y-axis shows frequency.
# 
# ### Analysis:
# 
# - **High Frequency of 17 to 20**: The highest concentration of final hand values occurs between **17** and **20**, with **20** being the most frequent. This pattern suggests that players commonly stop at values within this range, likely due to the favorable odds these hands provide without risking a bust.
# 
# - **Lower Frequency of 21**: Although **21** is the ideal hand in Blackjack, it appears less frequently than **20**. Achieving **21** often Blackjack or hitting exactly to reach this sum, which explains its lower occurrence. This indicates that while **21** is the goal, it is less commonly reached compared to slightly lower sums like **20**.
# 
# - **Frequent Busts (Values Above 21)**: The histogram shows a significant drop-off after **21**, with some frequency for values like **22** and above, indicating busts. Although players aim to avoid busting, these values still appear, showing that some players continue hitting beyond safe thresholds or encounter unlucky outcomes. However, the lower frequency of busts compared to final sums below **21** suggests players are cautious about avoiding these outcomes.
# 
# - **Low Frequency Below 16**: Final hands below **16** are relatively rare, as players generally aim to reach closer to **21** for a competitive hand. The scarcity of these lower values indicates that players are more likely to continue hitting until they achieve a hand close to **17** or higher, reinforcing typical Blackjack strategy.
# 
# ### Implications for Strategy:
# 
# This distribution indicates a strategic focus on achieving high yet safe hand values, particularly between **17** and **20**. The peak at **20** shows that players frequently settle on this strong hand to maximize winning chances without risking a bust. The rarity of final hands below **16 or above 21** underscores the emphasis on calculated risks, where players aim to approach **21** while minimizing the likelihood of busting. This histogram highlights the balance Blackjack players seek between risk and reward in their final hand decisions.


# ## **Stacked Bar Graph of Win/Loss/Push by Final Player Hand**


# Group by 'sumofcards' for the full player hand sum
win_loss_data = df.groupby('sumofcards')['winloss'].value_counts(normalize=True).unstack() * 100

# Define the color map for each outcome
color_map = {'Win': 'green', 'Loss': 'red', 'Push': 'grey'}

# Filter outcomes to only those available in the data
available_outcomes = [outcome for outcome in color_map if outcome in win_loss_data.columns]
colors = [color_map[outcome] for outcome in available_outcomes]

# Plotting the stacked bar chart
win_loss_data[available_outcomes].plot(kind='bar', stacked=True, figsize=(12, 6), color=colors)
plt.title('Win/Lose/Push Breakdown by Total Player Hand Sum')
plt.xlabel('Player Hand Sum')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=0)
plt.legend(title='Outcome')
plt.show()


# # **Win Rate for Initial Player Hand vs Final Player Hand**


# Calculate win rate by initial two-card sum
df['sum_first_two_cards'] = df['card1'] + df['card2']
df['winloss_binary'] = df['winloss'].apply(lambda x: 1 if x == 'Win' else 0)
win_rate_initial_hand = df.groupby('sum_first_two_cards')['winloss_binary'].mean() * 100

# Calculate win rate by final player hand sum (sum of all cards)
win_rate_final_hand = df.groupby('sumofcards')['winloss_binary'].mean() * 100

# Plot both win rates on the same line chart for comparison
plt.figure(figsize=(10, 6))
plt.plot(win_rate_initial_hand.index, win_rate_initial_hand.values, marker='o', color='blue', label='Initial Player Hand Win Rate')
plt.plot(win_rate_final_hand.index, win_rate_final_hand.values, marker='o', color='purple', label='Final Player Hand Win Rate')
plt.title('Comparison of Win Rate by Initial vs. Final Player Hand Sum')
plt.xlabel('Player Hand Sum')
plt.ylabel('Win Rate (%)')
plt.xticks(range(2, 27))
plt.grid(True)
plt.legend()
plt.show()


# This line chart compares the **win rates** of **initial** versus **final player hand sums** in Blackjack. The **blue line** represents the win rate based on the player’s initial two cards, while the **purple line** shows the win rate based on the final player hand sum after any hits or stands.
# 
# ### Analysis:
# 
# - **Initial vs. Final Hand Trends**: The win rate for initial hands (blue) gradually increases as the initial hand sum rises, peaking at higher values close to 21. The final hand win rate (purple) varies more significantly, with a sharp peak around 20 and a steep drop-off at 22, which represents a bust.
# 
# - **Sharp Increase for Final Hand at 20**: The final hand win rate spikes dramatically at **20**, indicating that this is one of the best hands to end with, providing a high probability of winning without the risk of busting. The final hand win rate at 21 is also high, but the steep drop to **0%** beyond 21 shows that busts eliminate any chance of winning.
# 
# - **Divergence in Mid-Range Hands (8-15)**: In the mid-range values from **8 to 15**, the final hand win rate fluctuates more than the initial hand win rate, indicating varying outcomes based on whether players choose to hit or stand. Lower final hand values (under 12) have low win rates due to insufficient competitiveness against the dealer’s likely hand, while final hands around 15-17 show relatively stable but modest win rates. Also, in the same range, the initial hands loses more than final hand because the player my decide to hit.
# 
# - **Convergence at High Values (17-21)**: As the hand sum approaches **17 to 21**, the win rates for both initial and final hands converge, showing similar probabilities of winning. This convergence indicates that players with strong initial hands in this range generally stand, resulting in a final hand sum that closely matches the initial value. These high sums are favorable for winning, consistent with Blackjack strategies that recommend standing on strong hands.
# 
# ### Strategic Implications:
# 
# This comparison highlights the importance of both initial and final hand values. While higher initial hands generally provide an advantage, the peak win rate at a final sum of 20 emphasizes the strategic decision to stand on strong hands close to 21. The steep decline in the final hand win rate beyond 21 shows the risk of busting, which negates any winning chance. The variability in mid-range final hands suggests that decisions to hit or stand in this range can greatly impact outcomes, reinforcing the need for strategic flexibility depending on the dealer's visible card.


# # **4) How does dealer's various hand influence the likelihood of the player winning?**


# # **Win Rate by Dealer’s First Card (Face Up)**


df['winloss_binary'] = df['winloss'].apply(lambda x: 1 if x == 'Win' else 0)

# Calculate win rate by dealer's first card (face up)
win_rate_by_dealer_card = df.groupby('dealcard1')['winloss_binary'].mean() * 100

# Plot the win rate by dealer's first card (face up)
plt.figure(figsize=(10, 6))
plt.plot(win_rate_by_dealer_card.index, win_rate_by_dealer_card.values, marker='o', color='Green')
plt.title('Win Rate by Dealer’s First Card (Face Up)')
plt.xlabel('Dealer First Card (Face Up)')
plt.ylabel('Win Rate (%)')
plt.xticks(range(1, 12))  # Adjust for the range of dealer cards
plt.grid(True)
plt.show()

# # **Lose Rate by Dealer’s First Card (Face Up)**
# 


df['winloss_binary'] = df['winloss'].apply(lambda x: 1 if x == 'Loss' else 0)

# Calculate win rate by dealer's first card (face up)
win_rate_by_dealer_card = df.groupby('dealcard1')['winloss_binary'].mean() * 100

# Plot the win rate by dealer's first card (face up)
plt.figure(figsize=(10, 6))
plt.plot(win_rate_by_dealer_card.index, win_rate_by_dealer_card.values, marker='o', color='Red')
plt.title('Lose Rate by Dealer’s First Card (Face Up)')
plt.xlabel('Dealer First Card (Face Up)')
plt.ylabel('Lose Rate (%)')
plt.xticks(range(1, 12))  # Adjust for the range of dealer cards
plt.grid(True)
plt.show()

# # **Push Rate by Dealer’s First Card (Face Up)**


df['winloss_binary'] = df['winloss'].apply(lambda x: 1 if x == 'Push' else 0)

# Calculate win rate by dealer's first card (face up)
win_rate_by_dealer_card = df.groupby('dealcard1')['winloss_binary'].mean() * 100

# Plot the win rate by dealer's first card (face up)
plt.figure(figsize=(10, 6))
plt.plot(win_rate_by_dealer_card.index, win_rate_by_dealer_card.values, marker='o', color='Grey')
plt.title('Push Rate by Dealer’s First Card (Face Up)')
plt.xlabel('Dealer First Card (Face Up)')
plt.ylabel('Push Rate (%)')
plt.xticks(range(1, 12))  # Adjust for the range of dealer cards
plt.grid(True)
plt.show()

# # **Win, Push, and Lose Rates by Dealer's First Card (Face Up)**


# Calculate win, lose, and push rates by dealer's first card
df['winloss_binary'] = df['winloss'].apply(lambda x: 1 if x == 'Win' else 0)
win_rate_by_dealer_card = df.groupby('dealcard1')['winloss_binary'].mean() * 100

df['winloss_binary'] = df['winloss'].apply(lambda x: 1 if x == 'Loss' else 0)
lose_rate_by_dealer_card = df.groupby('dealcard1')['winloss_binary'].mean() * 100

df['winloss_binary'] = df['winloss'].apply(lambda x: 1 if x == 'Push' else 0)
push_rate_by_dealer_card = df.groupby('dealcard1')['winloss_binary'].mean() * 100

# Plot all three rates on the same graph
plt.figure(figsize=(10, 6))
plt.plot(win_rate_by_dealer_card.index, win_rate_by_dealer_card.values, marker='o', color='green', label='Win Rate')
plt.plot(lose_rate_by_dealer_card.index, lose_rate_by_dealer_card.values, marker='o', color='red', label='Lose Rate')
plt.plot(push_rate_by_dealer_card.index, push_rate_by_dealer_card.values, marker='o', color='grey', label='Push Rate')
plt.title('Win, Lose, and Push Rates by Dealer’s First Card (Face Up)')
plt.xlabel('Dealer First Card (Face Up)')
plt.ylabel('Rate (%)')
plt.xticks(range(1, 12))  # Adjust for the range of dealer cards
plt.grid(True)
plt.legend()
plt.show()

# This chart shows the **win, lose, and push rates** for the player based on the dealer's first face-up card in Blackjack. The **green line** represents the player’s win rate, the **red line** indicates the lose rate, and the **gray line** shows the push rate.
# 
# ### Analysis:
# 
# - **Dealer Low Cards (1-6)**: When the dealer’s face-up card is between **1** and **6**, the player’s win rate is generally higher than the lose rate, peaking around **55%**. This is due to the dealer's increased chance of busting with these low cards, making it favorable for players to stay in the game and potentially win.
# 
# - **Dealer High Cards (7-11)**: As the dealer’s face-up card moves from **7** to **11**, the win and lose rates start to reverse. By the time the dealer’s face-up card reaches **8**, the lose rate surpasses the win rate, showing a clear disadvantage for the player. The lose rate increases significantly, reaching over **60%** when the dealer shows a **10** or **Ace**, which represents a strong starting position for the dealer.
# 
# - **Push Rate**: The push rate remains relatively low across all dealer cards, staying under **10%**. It increases slightly as the dealer’s card value rises, peaking at **8** and above. However, pushes remain a minor outcome compared to wins and losses.


# # **Player Bust Rate Given Dealer First Card (Face Up)**


# Convert 'plybustbeat' to numeric where 'Bust' is 1, otherwise 0
df['plybustbeat_numeric'] = df['plybustbeat'].apply(lambda x: 1 if x == 'Bust' else 0)

# Calculate the player bust rate by dealer's first card (face up)
player_bust_rate_by_dealer_card = df.groupby('dealcard1')['plybustbeat_numeric'].mean() * 100

# Plot the player bust rate by dealer's first card as a line graph
plt.figure(figsize=(10, 6))
plt.plot(player_bust_rate_by_dealer_card.index, player_bust_rate_by_dealer_card.values, marker='o', color='Green', label="Player's Bust Rate")
plt.title('Player Bust Rate Given Dealer First Card (Face Up)')
plt.xlabel('Dealer First Card (Face Up)')
plt.ylabel('Bust Rate (%)')
plt.xticks(range(1, 12), rotation=0)  # Assuming dealer cards range from Ace (1) to King (11)
plt.grid(True)
plt.legend()
plt.show()

# # **Dealer Bust Rate Given Dealer First Card (Face Up)**


# Convert 'dlbustbeat' to numeric where 'Bust' is 1, otherwise 0
df['dlbustbeat_numeric'] = df['dlbustbeat'].apply(lambda x: 1 if x == 'Bust' else 0)

# Calculate the dealer bust rate by dealer's first card
dealer_bust_rate = df.groupby('dealcard1')['dlbustbeat_numeric'].mean() * 100

# Plot the dealer bust rate as a line graph
plt.figure(figsize=(10, 6))
plt.plot(dealer_bust_rate.index, dealer_bust_rate.values, marker='o', color='Red', label="Dealer's Bust Rate")
plt.title('Dealer Bust Rate Given Dealer First Card (Face Up)')
plt.xlabel('Dealer First Card (Face Up)')
plt.ylabel('Bust Rate (%)')
plt.xticks(range(1, 12), rotation=0)  # Adjust for the range of dealer cards
plt.grid(True)
plt.legend()
plt.show()

# # **Dealer and Player Bust Rates by Dealer’s First Card (Face Up)**


# Convert bust indicators to numeric: 'Bust' as 1, others as 0
df['dlbustbeat_numeric'] = df['dlbustbeat'].apply(lambda x: 1 if x == 'Bust' else 0)
df['plybustbeat_numeric'] = df['plybustbeat'].apply(lambda x: 1 if x == 'Bust' else 0)

# Calculate bust rates by dealer's first card (face up) for both dealer and player
dealer_bust_rate = df.groupby('dealcard1')['dlbustbeat_numeric'].mean() * 100
player_bust_rate = df.groupby('dealcard1')['plybustbeat_numeric'].mean() * 100

# Plot both bust rates on the same graph
plt.figure(figsize=(10, 6))
plt.plot(dealer_bust_rate.index, dealer_bust_rate.values, marker='o', color='Red', label='Dealer Bust Rate')
plt.plot(player_bust_rate.index, player_bust_rate.values, marker='o', color='Green', label='Player Bust Rate')
plt.title('Dealer and Player Bust Rates by Dealer’s First Card (Face Up)')
plt.xlabel('Dealer First Card (Face Up)')
plt.ylabel('Bust Rate (%)')
plt.xticks(range(1, 12), rotation=0)  # Assuming dealer cards range from Ace (1) to King (11)
plt.grid(True)
plt.legend()
plt.show()

# This chart compares the **dealer and player bust rates** based on the **dealer’s first card (face up)** in Blackjack. The **red line** represents the dealer’s bust rate, and the **green line** shows the player’s bust rate.
# 
# ### Analysis:
# 
# - **Dealer Bust Rate (1-6)**: The dealer’s bust rate is significantly higher when their first card is between **1** and **6**, peaking around **42%**. This is because the dealer has a greater chance of going over 21 with low starting values, especially since they are required to hit until reaching at least 17. The high bust rate for these cards provides a strong advantage for the player.
# 
# - **Dealer Bust Rate Decline (7-11)**: As the dealer’s first card moves from **7** to **11**, the bust rate declines sharply, approaching **0%** at **10** and **11** (Ace). These high-value cards put the dealer in a safer position to reach a competitive hand without busting, reducing the risk for the dealer and shifting the game’s odds in their favor.
# 
# - **Player Bust Rate**: The player’s bust rate remains relatively low across all dealer first cards, staying mostly below **30%**. It peaks at **30%** when the dealer shows an Ace but remains consistently lower than the dealer’s bust rate when the dealer’s card is **1-6**. This suggests that players are more conservative in these scenarios, possibly opting to stand more frequently to avoid the risk of busting, especially when the dealer’s hand is weak.
# 
# ### Key Insight:
# 
# The high dealer bust rate for low initial cards (1-6) provides a strategic edge for the player, while the player’s bust rate remains comparatively stable, emphasizing the advantage of playing conservatively when the dealer shows a weak hand. Conversely, when the dealer shows strong cards (10 or Ace), the game shifts in the dealer's favor as their bust probability drops.


# # **Linear Regression: Predicting Player Bust Rate by Dealer Face-Up Card**


# Convert bust outcomes to binary values where 'Bust' is 1, others are 0
df['bust_binary'] = df['plybustbeat'].apply(lambda x: 1 if x == 'Bust' else 0)

# Group by dealer's face-up card to calculate the average player bust rate for each card
bust_rate_by_dealer_card = df.groupby('dealcard1')['bust_binary'].mean() * 100

# Prepare the data for numpy-based linear regression
X = bust_rate_by_dealer_card.index.values  # Dealer's face-up card
y = bust_rate_by_dealer_card.values        # Player bust rate

# Perform linear regression using numpy
X_mean = np.mean(X)
y_mean = np.mean(y)
numerator = np.sum((X - X_mean) * (y - y_mean))
denominator = np.sum((X - X_mean) ** 2)
slope = numerator / denominator
intercept = y_mean - (slope * X_mean)

# Calculate the predicted values for plotting
predictions = slope * X + intercept

# Print the slope and intercept of the regression line
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")

# Plot the regression line with the data
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Bust Rate Data')
plt.plot(X, predictions, color='red', label='Regression Line')
plt.xlabel('Dealer Face-Up Card')
plt.ylabel('Player Bust Rate (%)')
plt.title('Linear Regression: Predicting Player Bust Rate by Dealer Face-Up Card')
plt.xticks(np.arange(min(X), max(X) + 1, 1))  # Set x-axis increments to 1
plt.legend()
plt.grid(True)
plt.show()

# This linear regression plot models the relationship between the **dealer’s face-up card** and the **player's bust rate**. The **blue points** represent actual bust rate data based on the dealer’s first card, while the **red line** represents the linear regression line with a slope of **2.34** and an intercept of **1.97**.
# 
# ### Analysis:
# 
# - **Positive Correlation**: The positive slope of **2.34** suggests that as the dealer’s face-up card increases in value, the player’s likelihood of busting also increases. This correlation indicates that players are more likely to take additional risks (and potentially bust) when the dealer shows a high-value card, as they attempt to reach a stronger hand to compete.
# 
# - **Intercept Insight**: The intercept (**1.97**) indicates a baseline bust rate when the dealer’s card is minimal (hypothetically close to zero). This intercept has limited practical meaning.
# 
# - **Rising Bust Rate with Dealer Strength**: As the dealer’s card reaches higher values (especially 10 or Ace), the player’s bust rate is higher. This aligns with Blackjack strategy, where players often feel compelled to hit more aggressively when facing a strong dealer hand, increasing their risk of busting.
# 
# ### Conclusion:
# 
# This regression analysis shows that players are more prone to busting when the dealer’s face-up card is high. This tendency reflects the player’s strategic adjustments in response to dealer strength.


# # **Heatmap of Player Win Rate by Initial Hand and Dealer Face-Up Card**


# Convert win/loss outcomes to binary values where 'Win' is 1, others are 0
df['win_binary'] = df['winloss'].apply(lambda x: 1 if x == 'Win' else 0)

# Calculate the sum of the player's first two cards
df['sum_first_two_cards'] = df['card1'] + df['card2']

# Create a pivot table for win rate based on player's initial hand and dealer's face-up card
win_rate = df.pivot_table(
    values='win_binary',
    index='sum_first_two_cards',
    columns='dealcard1',
    aggfunc='mean'
) * 100  # Convert to percentage

# Plot the heatmap with a red color scheme
plt.figure(figsize=(12, 8))
sns.heatmap(win_rate, annot=True, fmt=".1f", cmap="Reds", cbar_kws={'label': 'Win Rate (%)'})
plt.title('Heatmap of Player Win Rate by Initial Hand and Dealer Face-Up Card')
plt.xlabel('Dealer Face-Up Card')
plt.ylabel('Player Initial Hand Sum')
plt.show()

# This heatmap illustrates the **player's win rate** based on the **initial player hand sum** (y-axis) and the **dealer's face-up card** (x-axis). Darker shades indicate higher win rates, while lighter shades indicate lower win rates.
# 
# ### Analysis:
# 
# - **High Player Win Rate at Strong Initial Hands (20-21)**: The win rate is highest when the player’s initial hand sum is **20** or **21**, especially when the dealer’s face-up card is a low value (1-6). At these values, the win rate exceeds **90%**, emphasizing the advantage of a near-perfect or perfect starting hand.
# 
# - **Impact of Dealer’s Strong Face-Up Cards (10 and 11)**: When the dealer’s face-up card is **10** or **11** (Ace), the player’s win rate is generally lower across all initial hand sums. Even with an initial hand sum of **20**, the win rate drops to around **57.4%** against a dealer Ace. This reflects the increased likelihood that the dealer will achieve a high final hand or a Blackjack, reducing the player's winning probability.
# 
# - **Moderate Initial Hands (12-16)**: For initial hand sums in the range of **12 to 16**, the player’s win rate is relatively moderate, around **30-45%**, when the dealer shows a low to mid-range face-up card (1-6). However, these win rates sharply decrease as the dealer’s face-up card value rises, highlighting the difficulty of winning with mid-range hands against a strong dealer position.
# 
# - **Low Win Rates with Weak Initial Hands (2-8)**: Player win rates are particularly low when the initial hand sum is between **2 and 8**, regardless of the dealer’s face-up card. This range shows the least favorable outcomes for the player, with win rates typically below **40%** and dropping significantly when the dealer’s face-up card is a high value.
# 
# ### Strategic Insights:
# 
# This heatmap depicts the advantage of a high initial hand and the challenge of competing against strong dealer cards. Players have the best chances with initial hands of **20** or **21**, particularly when the dealer’s card is weak, which highlights the strategic benefit of standing on strong hands. Conversely, mid to low initial hand sums require careful decision-making, as they present lower win rates, especially against a high dealer card.


# # **Heatmap of Player Bust Probability by Initial Hand and Dealer Face-Up Card**


# Convert bust outcomes to binary values where 'Bust' is 1, others are 0
df['bust_binary'] = df['plybustbeat'].apply(lambda x: 1 if x == 'Bust' else 0)

# Calculate the sum of the player's first two cards
df['sum_first_two_cards'] = df['card1'] + df['card2']

# Create a pivot table for bust probability based on player's initial hand and dealer's face-up card
bust_probability = df.pivot_table(
    values='bust_binary',
    index='sum_first_two_cards',
    columns='dealcard1',
    aggfunc='mean'
) * 100  # Convert to percentage

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(bust_probability, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Bust Probability (%)'})
plt.title('Heatmap of Player Bust Probability by Initial Hand and Dealer Face-Up Card')
plt.xlabel('Dealer Face-Up Card')
plt.ylabel('Player Initial Hand Sum')
plt.show()

# This heatmap displays the **player's bust probability** based on the **initial player hand sum** (y-axis) and the **dealer's face-up card** (x-axis). Darker shades represent higher bust probabilities, while lighter shades indicate lower bust probabilities.
# 
# ### Analysis:
# 
# - **High Bust Probability for Low Initial Hands (2-8)**: When the player's initial hand sum is low (2-8), the bust probability is generally higher, particularly when the dealer’s face-up card is **1** or **2**. This trend suggests that players with weak starting hands tend to take more risks (i.e., hit more), leading to a higher chance of busting.
# 
# - **Moderate Initial Hands (12-16)**: For initial hand sums between **12 and 16**, bust probabilities are notably high, especially when the dealer’s face-up card is a mid-range value (4-6). This aligns with standard Blackjack strategy, where players often hit on these values to improve their hand but are at a significant risk of busting due to the increased likelihood of going over 21.
# 
# - **Low Bust Probability for Strong Hands (17-21)**: When the player’s initial hand sum is between **17 and 21**, the bust probability is effectively **0%** across all dealer face-up cards, as players are likely to stand on these strong hands. This minimizes the bust risk and highlights the advantage of starting with a higher hand total.
# 
# ### Key Insights:
# 
# This heatmap reveals that bust probability is highly dependent on both the player’s initial hand strength and the dealer’s visible card. Low initial hands prompt riskier decisions, resulting in higher bust rates, while strong hands (17 and above) significantly reduce the bust probability as players tend to stand.


# # **What is the Probability that the Player Lose Given the Player Gets 20? (Dealer Gets 21)**


# Filter the dataset to include only hands where the player's final hand sum is 20
player_20 = df[df['sumofcards'] == 20]

# Calculate the probability that the dealer's final hand is 21 given the player has 20
prob_house_21_given_player_20 = (player_20['sumofdeal'] == 21).mean() * 100

# Print the probability
print(f"The Probability that the Player Loses Given the Player Gets 20: {prob_house_21_given_player_20}%")

# This analysis reveals that even with a strong hand sum of **20**, the player faces a **12.08%** chance of losing if the dealer achieves a **21**. While **20** is typically a highly favorable position, this probability highlights the unpredictability in Blackjack, where even optimal hands aren’t guaranteed wins. This result underscores the risk players face, as the dealer’s ability to reach **21** can occasionally overturn the player's win.


# # **5) Conclusion**


# Through the analysis of player and dealer hand interactions, probabilities, and outcomes in Blackjack, we explored the factors that influence a player’s likelihood of winning, losing, or busting. By examining player win rates, bust probabilities, and the impact of both initial player hands and dealer face-up cards, we gathered insights into optimal and suboptimal scenarios. This data-driven approach provided a clear view of how initial hand strength and dealer card visibility shape the game's strategic landscape. Strong initial hands, especially close to 21, significantly increase the player’s win probability, while lower initial hand sums tend to prompt riskier moves and higher bust rates.
# 
# The analysis highlights several favorable scenarios in which the player is more likely to win in Blackjack. One of the clearest indicators of a good winning scenario is when the player’s initial hand sum is **20 or 21**; in these cases, the player’s win rate reaches its peak, as these strong starting totals minimize the need for additional risky hits. Similarly, when the dealer’s face-up card is **4, 5, or 6**, the player’s chances of winning increase significantly, as these cards put the dealer at a higher bust risk due to the likelihood of forced hits that lead to exceeding **21**. Additionally, if the player’s initial hand is between **12 and 16** and the dealer shows a low or mid-range card (especially **4** through **6**), standing can be a wise choice as the dealer is more likely to bust, shifting the odds in favor of the player. These favorable conditions shows how strategic decisions based on both the player’s total and the dealer’s visible card can greatly improve the chances of winning, demonstrating the importance of situational awareness in Blackjack strategy.
# 
# Our findings emphasize the complexity of decision-making in Blackjack, where each combination of player and dealer cards presents unique win or bust probabilities. The heatmaps and regression analyses showed that a player’s chance of winning depends heavily on both their starting hand and the dealer's face-up card. This highlights the importance of calculated risk in hitting or standing based on the player’s current hand total relative to the dealer’s visible card.
# 
# In conclusion, the insights drawn from this dataset offer a refined understanding of Blackjack’s underlying probabilities. Players aiming to maximize their winning potential can benefit from adopting a strategy that accounts for both their hand total and the dealer’s face-up card. However, while probability provides guidance, Blackjack retains elements of unpredictability, where calculated risks may not always lead to success.