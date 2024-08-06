#!/usr/bin/env python
# coding: utf-8

# # AIRLINE CUSTOMER LOYALTY PROGRAM
# **Customer loyalty program data from Northern Lights Air (NLA), a fictitious airline based in Canada. In an effort to improve program enrollment, NLA ran a promotion between Feb - Apr 2018. Dataset includes loyalty program signups, enrollment and cancellation details, and additional customer information.**

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# CSV files into Pandas DataFrames
df_cfa=pd.read_csv(r"C:\Users\Win11 Pro\Desktop\airline loyalty program\Customer Flight Activity.csv")
df_clh=pd.read_csv(r"C:\Users\Win11 Pro\Desktop\airline loyalty program\Customer Loyalty History.csv")
df_cfa


# In[3]:


df_cfa.skew()


# In[4]:


df_clh


# In[5]:


# first few rows of each DataFrame to understand their structure
df_cfa.head()


# In[6]:


df_cfa.tail()


# In[7]:


df_cfa.info()


# In[8]:


df_cfa.describe()


# In[9]:


df_clh.describe()


# In[10]:


df_cfa.describe().columns


# In[11]:


df_clh.describe().columns


# In[12]:


list(set(df_clh.columns.tolist())-set(df_clh.describe().columns))


# In[13]:


pd.isnull(df_cfa).sum()


# In[14]:


pd.isnull(df_clh).sum()


# # Mean

# In[15]:


df_cfa.isnull().sum()/len(df_cfa)


# In[16]:


df_clh.isnull().sum()/len(df_clh)


# In[17]:


round((df_cfa.isnull().sum()/len(df_cfa)*100),3) 


# In[18]:


round((df_clh.isnull().sum()/len(df_clh)*100),3)


# In[19]:


def nullvalues(dataframe):
    return round((dataframe.isnull().sum()/len(dataframe)*100),3)
nullvalues(df_cfa)


# In[20]:


nullvalues(df_clh)


# In[21]:


round((df_clh.isnull().sum()/len(df_clh)*100),3).sort_values(ascending=False)


# In[22]:


null=nullvalues(df_cfa)[nullvalues(df_cfa)>40]
print(null)


# In[23]:


null1=nullvalues(df_clh)[nullvalues(df_clh)>40]
print(null1)


# # SKEWNESS AND IT'S TREATMENT

# In[24]:


## making a sample dataset to check and skewness on CUSTOMER FLIGHT ACTIVITY:
df_sample=pd.read_csv((r"C:\Users\Win11 Pro\Desktop\airline loyalty program\Customer Flight Activity.csv"))
df_sample


# In[25]:


DATA=df_sample.skew()
DATA


# In[58]:


from scipy.stats import skew
from scipy.stats import boxcox
plt.figure(figsize=(10, 6))
plt.hist(DATA, bins=30, edgecolor='k',alpha=0.7)
plt.title('Histogram of Skewed Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()


# In[56]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, boxcox

# Example DataFrame
data = {
    'Loyalty Number': df_cfa["Loyalty Number"],
    'Year': df_cfa["Loyalty Number"],
    'Month':  df_cfa["Month"],
    'Total Flights':  df_cfa["Total Flights"],
    'Distance':  df_cfa["Distance"],
    'Points Accumulated':  df_cfa["Points Accumulated"],
    'Points Redeemed':  df_cfa["Points Redeemed"],
    'Dollar Cost Points Redeemed':  df_cfa["Dollar Cost Points Redeemed"]
}

df = pd.DataFrame(data)

# Identify skewed columns
skewed_columns = df.apply(skew).sort_values(ascending=False)
skewed_columns = skewed_columns[abs(skewed_columns) > 1]

# Function to apply transformations
def transform_skewed(df, skewed_columns):
    df_transformed = df.copy()
    for col in skewed_columns.index:
        if any(df_transformed[col] <= 0):
            # If there are non-positive values, apply log(x + 1)
            df_transformed[col] = np.log1p(df_transformed[col])
        else:
            # If all values are positive, apply log transformation
            df_transformed[col] = np.log(df_transformed[col])
            # Alternatively, you could use Box-Cox if you know your data supports it
            # df_transformed[col], _ = boxcox(df_transformed[col])
    return df_transformed

# Apply transformations
df_transformed = transform_skewed(df, skewed_columns)
def plot_histograms(df, title):
    cols = df.columns
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(cols):
        plt.subplot(3, 3, i+1)
        sns.histplot(df[col], kde=True)
        plt.title(f'{col} (skew: {skew(df[col]):.2f})')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# In[28]:


skewed_columns


# In[29]:


plot_histograms(df, "Original Data")


# In[30]:


plot_histograms(df_transformed, "Transformed Data")


# # EDA on Customer Loyalty History

# In[31]:


enrollments_during_promo=df_clh[(df_clh['Enrollment Year'] == 2018) & (df_clh['Enrollment Month'].between(2, 4))].shape[0]
enrollments_during_promo


# In[32]:


## dropping null values from 'Cancellation Year', 'Cancellation Month'.
no_cancellations=df_clh.dropna(subset=['Cancellation Year', 'Cancellation Month'])
cancellations_during_promo=no_cancellations[(no_cancellations['Cancellation Year'] == 2018) & 
(no_cancellations['Cancellation Month'].between(2, 4))].shape[0]
cancellations_during_promo


# In[33]:


net_membership_change_during_promo = enrollments_during_promo - cancellations_during_promo

{
    "enrollments During Promo ": enrollments_during_promo,
    "Cancellations During Promo": cancellations_during_promo,
    "Net Membership Change During Promo": net_membership_change_during_promo
}


# In[34]:


#total number of enrollments by year and month
enrollment_trend = df_clh.groupby(['Enrollment Year', 'Enrollment Month']).size().reset_index(name='Enrollments')
enrollment_trend


# In[35]:


plt.figure(figsize=(12, 4)) 
sns.lineplot(data=enrollment_trend, x='Enrollment Year', 
             y='Enrollments', estimator='sum',errorbar=None, marker='o')
plt.title('Enrollments over years')
plt.ylabel('Number of Enrollments')

    
plt.show()


# # This indicates that the promotional campaign had a positive impact on the loyalty program's membership, with a significant net increase in memberships during the promotional period.

# # 
# 
# 
# # Demographic Breakdown of Loyalty Members
# 
# SINGLE VARIATE ANALYSIS 

# In[36]:


# the success of the campaign among different demographics
# enrollments during the promotional period
enrollments_during_promo = df_clh[(df_clh['Enrollment Year'] == 2018) & (df_clh['Enrollment Month'].between(2, 4))]
enrollments_during_promo


# In[37]:


enrollments_by_marital_status = df_clh['Marital Status'].value_counts()
enrollments_by_marital_status 


# In[38]:


enrollments_by_gender = df_clh['Gender'].value_counts()
enrollments_by_gender


# In[39]:


enrollments_by_education = df_clh['Education'].value_counts()
enrollments_by_education


# In[40]:


# Gender Distribution Among Loyalty Members

plt.figure(figsize=(6,4))
sns.countplot(data=df_clh, x='Gender', palette='coolwarm')
plt.title('Gender Distribution Among Loyalty Members')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.tight_layout()

plt.show()


# In[41]:


# Education Distribution Among Loyalty Members

plt.figure(figsize=(10, 6))
sns.countplot(data=df_clh, x='Education', palette='coolwarm')
plt.title('Education Distribution Among Loyalty Members')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.show()


# In[42]:


# Marital Status Distribution Among Loyalty Members

plt.figure(figsize=(10,3))
sns.countplot(data=df_clh, x='Marital Status', order=df_clh['Marital Status'].value_counts().index , palette='magma')
plt.title('Marital Status Distribution Among Loyalty Members')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.show()


# # **Summer 2018 Flight Activity and Loyalty Rewards Analysis**
# This analysis investigates the relationship between flight distances and loyalty points accumulated by Northern Lights Air (NLA) loyalty members during the summer months of 2018. By examining the distribution of flights booked and visualizing the correlation between the total distance flown and points earned, we gain insights into member engagement levels and the effectiveness of promotional campaigns in encouraging travel.

# # **The impact of the campaign on booked flights during the summer months (June, July, August)**
# 
# BI VARIATE ANALYSIS

# In[43]:


# flight activity for summer months

summer_months = [6, 7, 8]
flight_activity_summer= df_cfa[(df_cfa['Year'] == 2018) & (df_cfa['Month'].isin(summer_months))]

flight_activity_summer


# In[44]:


# Total flight activity by loyalty number to check the overall flight activity during the summer

total_flight_activity_summer = flight_activity_summer.groupby('Loyalty Number').agg(
    Total_Flights_Summer=('Total Flights', 'sum'),
    Total_Distance_Summer=('Distance', 'sum'),
    Points_Accumulated_Summer=('Points Accumulated', 'sum')
).reset_index()
total_flight_activity_summer


# In[45]:


total_flights_summer= total_flight_activity_summer['Total_Flights_Summer'].sum()
total_distance_summer = total_flight_activity_summer['Total_Distance_Summer'].sum()
total_points_summer = total_flight_activity_summer['Points_Accumulated_Summer'].sum()

{
    "Total Flights During Summer": total_flights_summer,
    "Total Distance Flown During Summer": total_distance_summer,
    "Total Points Accumulated During Summer": total_points_summer
}


# In[46]:


## histogram to show the distribution of the number of flights booked by individual members during summer 

plt.figure(figsize=(8, 5))
sns.histplot(total_flight_activity_summer['Total_Flights_Summer'], bins=35, kde=False,color="PUrple")
plt.title('Distribution of Flights Booked by Loyalty Members During Summer')
plt.xlabel('Total Flights Booked')
plt.ylabel('Number of Members')
plt.show()


# In[47]:


# Scatter plot for the relationship between distance flown and points accumulated during summer 

plt.figure(figsize=(8, 5))
sns.scatterplot(data=total_flight_activity_summer, x='Total_Distance_Summer', y='Points_Accumulated_Summer', alpha=0.6,color="green")
plt.title('Relationship Between Distance Flown and Points Accumulated During Summer')
plt.xlabel('Total Distance Flown (miles)')
plt.ylabel('Points Accumulated')
plt.tight_layout()

plt.show()


# # The scatter plot reveals the relationship between the total distance flown and the points accumulated by loyalty members during the summer of 2018. The plot indicates a positive correlation between these two metrics: as the distance flown increases, the points accumulated also tend to increase. This suggests that members who flew longer distances accrued more loyalty points, aligning with expectations for loyalty programs where rewards are often proportional to travel activity.
# 
# ### **Insights:**
# 
# **Flight Activity Distribution:** The majority of loyalty members booked a relatively low number of flights, with a small subset booking many flights. This indicates diverse engagement levels within the loyalty program.
# 
# **Distance vs. Points:** The positive correlation between distance flown and points accumulated confirms that more active travelers, in terms of distance, benefited more in terms of loyalty rewards. This could incentivize members to book longer or more flights to accumulate points faster.
# 
# **Promotional Campaign Influence:** The significant level of flight activity and points accumulation during the summer months post-campaign suggests that the promotional efforts may have successfully encouraged increased travel activity among members. This could be particularly true for members motivated by the opportunity to earn more points through increased flight bookings.

# In[48]:


df_clh['Enrollment date'] = df_clh['Enrollment Year'].astype(str) + '-' + df_clh['Enrollment Month'].astype(str).str.zfill(2)


# In[49]:


df_clh['Enrollment date']


# # **Merging customer flight activity and customer loyalty history**
# 

# In[50]:


flight_activity = pd.merge(df_cfa,df_clh[['Loyalty Number', 'Enrollment date']], on='Loyalty Number')
flight_activity


# In[51]:


# Aggregate flight activity by enrollment date
agg_flight_activity = flight_activity.groupby('Enrollment date').agg(
    Average_Flights_Booked=('Total Flights', 'mean'),
    Average_Points_Accumulated=('Points Accumulated', 'mean')
).reset_index()
agg_flight_activity


# In[52]:


flight_activity['Enrollment date'] = pd.to_datetime(flight_activity['Enrollment date'], format='%Y-%m')

flight_activity = flight_activity.sort_values('Enrollment date')
flight_activity


# In[53]:


# The average flights and points 
plt.figure(figsize=(10, 6))

# Plot for average flights booked
plt.subplot(1, 2, 1)
sns.lineplot(data=agg_flight_activity, x='Enrollment date', y='Average_Flights_Booked', marker='o')
plt.title('Average Flights Booked by Enrollment date')
plt.xlabel('Enrollment date')
plt.ylabel('Average Flights Booked')


# Plot for average points accumulated

plt.subplot(1, 2, 2)
sns.lineplot(data=agg_flight_activity, x='Enrollment date', y='Average_Points_Accumulated', marker='o', color='orange')
plt.title('Average Points Accumulated by Enrollment date')
plt.xlabel('Enrollment date')
plt.ylabel('Average Points Accumulated')
plt.tight_layout()
plt.show()


# # Analysis of Flight Activity by Province

# In[54]:


# Merge the 'Customer Flight Activity' dataset with the 'Customer Loyalty History' dataset on 'Loyalty Number'
flight_data = pd.merge(df_cfa, df_clh[['Loyalty Number', 'Province']], on='Loyalty Number')

# Aggregate total flights by province
total_flights_by_province = flight_data.groupby('Province').agg(Total_Flights=('Total Flights', 'sum')).reset_index()

total_flights_by_province_sorted = total_flights_by_province.sort_values(by='Total_Flights', ascending=True)

total_flights_by_province_sorted


# In[55]:


plt.figure(figsize=(10, 6))
sns.barplot(data=total_flights_by_province_sorted, x='Total_Flights', y='Province', palette='coolwarm')
plt.title('Total Flights by Province')
plt.xlabel('Total Flights')
plt.ylabel('Province')
for index, value in enumerate(total_flights_by_province_sorted['Total_Flights']):
    plt.text(value, index, str(value))
plt.tight_layout()
plt.show()


# 
# # Project Overview
# 
# This project focused on analyzing the loyalty program data from Northern Lights Air (NLA), a fictitious airline based in Canada. The analysis covered the impact of a promotional campaign on loyalty program enrollments, demographic adoption, flight bookings during summer, and flight activity by province. Key areas of analysis included:
# 
# **Promotional Campaign Impact:** Evaluating the success of a campaign aimed at increasing loyalty program memberships.
# 
# **Demographic Adoption:** Analyzing which demographics were more inclined to enroll in the loyalty program during the campaign.
# 
# **Summer Flight Bookings:** Assessing the effect of the promotional campaign on the number of flights booked during the summer months.
# 
# **Flight Activity by Province:** Understanding regional differences in flight bookings among loyalty program members.

# # Insights
# 
# **Successful Promotion:** The promotional campaign led to a significant net increase in loyalty program memberships, indicating its effectiveness in attracting new members.
# 
# **Demographic Patterns:** The campaign saw higher adoption among certain demographics, particularly among individuals with a Bachelor's degree and those who are married, suggesting that future campaigns could be tailored more specifically to target these groups.
# 
# **Increased Summer Activity:** There was a notable increase in flight bookings during the summer months following the campaign, suggesting that the promotion not only boosted loyalty program enrollments but also encouraged immediate travel activity.
# 
# **Regional Flight Activity Variations:** The analysis of flight activity by province revealed significant regional variations, with Ontario and British Columbia leading in total flights. This points to a strong market presence in these provinces and potential areas for growth in others.

# # Strategic Considerations
# 
# **Tailored Marketing Strategies:** Future promotional campaigns can be designed to target demographics showing lower enrollment rates, leveraging insights from the campaign analysis to increase effectiveness.
# 
# **Seasonal Promotions:** Given the increase in summer flight bookings, NLA could consider seasonal promotions to capitalize on periods of high travel demand, possibly extending special offers to loyalty program members to encourage booking.
# 
# **Enhanced Focus on Key Markets:** Ontario and British Columbia, being the highest in flight activity, should continue to receive focused attention. Strategies could include enhancing route availability, offering exclusive loyalty program benefits, or partnering with local businesses to increase engagement.
# 
# **Growth Opportunities in Emerging Markets:** Provinces with moderate to low flight activities represent untapped potential. NLA could explore introducing new routes, customizing loyalty rewards to regional preferences, or conducting market research to understand barriers to engagement.
# 
# **Retention Strategies:** While not directly analyzed, the importance of understanding cancellation trends suggests the need for strategies aimed at retaining members, such as improving program benefits, personalized communication, and addressing feedback.
