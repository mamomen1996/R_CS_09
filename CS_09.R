# Case-Study Title: Customers RFM Clustering (Market Segmentation based on Behavioral Approach)
# Data Analysis methodology: CRISP-DM
# Dataset: Iranian online e-commerce platform's customers transactions data in first 4 months of year 1398 (from 1398/01/01 to 1398/04/31)
# Case Goal: Detect and Segment similar customers of e-commerce platform business (Customer Segmentation using RFM model)


### Required Libraries ----
install.packages('factoextra')
install.packages('ggplot2')
install.packages('cluster')
library('factoextra')
library('ggplot2')
library('cluster')


### Read Data from File ----
data <- read.csv('CS_09.csv', header = T)
dim(data)  # 40537 records, 5 variables


### Step 1: Business Understanding ----
 # know business process and issues
 # know the context of the problem
 # know the order of numbers in the business


### Step 2: Data Understanding ----
### Step 2.1: Data Inspection (Data Understanding from Free Perspective) ----
## Dataset variables definition
colnames(data)

#order_id       -> ID of customer's order (Transaction unique ID)
#created_ts     -> Date of ordering in EN (date of Transaction)
#shamsy_date    -> Date of ordering in FA
#customer_id    -> ID of customer
#total_purchase -> sum of purchase for ordering transaction (Transaction total payment Price in Rials)


### Step 2.2: Data Exploring (Data Understanding from Statistical Perspective) ----
## Overview of Dataframe
class(data)
head(data)
tail(data)
str(data)
summary(data)
sum(is.na(data$total_purchase))  # has NA?

## Analyze daily demand (analyze number of transactions per day):
data$date <- as.Date(data$created_ts, '%m/%d/%Y')  # convert Character to Date
class(data$date)
daily_demand <- table(data$date)  # count records (transactions|demand) in each Category (day|date)
head(daily_demand)
mean(daily_demand)  # average daily transactions of this Business: 327 transactions in a day
plot(daily_demand, type = 'l')  # line chart: daily-demand changes through time
#there is a seasonal pattern in daily-demand of this business
 #we have demand fall in weekends and holidays in this Business (because of Nature of demand)
 #we have a pick in last week because of marketing campaign


### Step 3: Data PreProcessing ----
## Create RFM dataset
#we want to use RFM model for our Cluestering, so we need to prepare Recency-Frequency-Monetary for every customer at this analysis-time-range

# Frequency: number of purchases per customer at analysis-time-range
customer_f <- as.data.frame(aggregate(data$order_id, list(data$customer_id), length))  # count number of transactions per customer_id in 4 months
colnames(customer_f) <- c('customer_id', 'freq')
head(customer_f)
length(customer_f$customer_id)  # 14964

hist(customer_f$freq, breks = 50)
summary(customer_f$freq)

# Recency: how long it passed from a customer's last purchase time?
tail(data)
r_date <- as.Date("07/23/2019", format = "%m/%d/%Y")  # reference date
customer_r <- as.data.frame(aggregate(data$date, list(data$customer_id), max))  # last date per each customer_id
colnames(customer_r) <- c('customer_id', 'last_date')
head(customer_r)  # last transaction date per customer_id

customer_r$recency <- as.numeric(r_date - customer_r$last_date)  # difference between two date in days
head(customer_r)  # passed days from each customer's last purchase?

hist(customer_r$recency, breaks = 50)
summary(customer_r$recency)

# Monetary: total purchase per customer
customer_m <- as.data.frame(aggregate(data$total_purchase, list(data$customer_id), sum))  # total purchase per customer_id in 4 months in Rials
colnames(customer_m) <- c('customer_id', 'monetary')
head(customer_m)

hist(customer_m$monetary, breaks = 50)
summary(customer_m$monetary)

# RFM dataset for Customers
df <- merge(customer_f, customer_r, 'customer_id')  # merge two dataframe based-on 'customer_id' column
head(df)

rfm_customer <- merge(df, customer_m, 'customer_id')
head(rfm_customer)

rfm_customer <- rfm_customer[, -3]  # remove 'last_date'
head(rfm_customer)  # R-F-M per each customer

rownames(rfm_customer) <- rfm_customer$customer_id  # assign customer ids to row names
rfm_customer <- rfm_customer[,-1]  # remove 'customer_id'
head(rfm_customer)

plot(rfm_customer$freq, rfm_customer$recency)  # there is not any pattern
plot(rfm_customer$freq, rfm_customer$monetary)  # there is a Strong positive linear-relationship between two features
cor(rfm_customer$freq, rfm_customer$monetary)  # high correlation

rfm_customer_2 <- rfm_customer[,c('freq', 'recency')]  # remove 'monetary' column from our Clustering features
head(rfm_customer_2)

hist(rfm_customer$freq, breaks = 50)  # so skewed data
hist(log10(rfm_customer$freq), breaks = 50)  # still skewed log(data)

hist(rfm_customer$recency, breaks = 50)
hist(log10(rfm_customer$recency), breaks = 50)

#Scale features
rfm_customer_2 <- scale(rfm_customer_2)  # bring data around 0
head(rfm_customer_2)
summary(rfm_customer_2)
class(rfm_customer_2)

hist(rfm_customer_2[,1], breaks = 50) #skewed data
hist(rfm_customer_2[,2], breaks = 50) #skewed data


### Step 4: Modeling ----
# Model 1: K-Means
#First try
set.seed(123)
seg_km1 <- kmeans(rfm_customer_2, centers = 5)  # 5 clusters
seg_km1

#Results
seg_km1$cluster  # each observation (customer) is in which cluster?
table(seg_km1$cluster)  # each cluster's population
km_res1 <- as.data.frame(seg_km1$cluster)
km_res1$customer_id <- rownames(km_res1)  # add 'customer_id' column again
colnames(km_res1) <- c('cluster', 'customer_id')
head(km_res1)

#add every customer's cluster label in run_1_k-means:
rfm_customer$km1 <- km_res1[,'cluster']

head(rfm_customer)  # 'km1': cluster label of observation at k-means 1th-run

aggregate(rfm_customer[,c(1:3)], list(rfm_customer$km1), mean)  # mean of R, F, M for each cluster
#give sense about customers in each Cluster
 #cluster 1: min buy-freq in 4-month and max buy-recency and min buy-monetary -> churned customers
 #cluster 2: max buy-freq in 4-month and low buy-recency and max buy-monetary -> valuable (loyal) customers
 #cluster 3: low buy-freq in 4-month and low buy-recency -> probably the customers which are new-added with campaign -> goal for work-on them to bring them to loyal customers
 #cluster 4: high buy-freq in 4-month and low buy-recency -> our good customers
 #cluster 5: low buy-freq in 4-month and medium buy-recency -> comeback them via a marketing game

table(rfm_customer$km1)

#visualize clusters
ggplot(data = rfm_customer, aes(x = freq, y = recency, color = factor(km1))) +
	geom_point() +
	ggtitle('kmeans - iter_1')

#Second try (because skewed data and k-means weakness)
set.seed(11234)
seg_km2 <- kmeans(rfm_customer_2, centers = 5)

#Results
table(seg_km2$cluster)  # different cluster sizes with km1 -> different clustering results
table(seg_km1$cluster)  # Note: do not care to cluster labels; because they can vary in each run

km_res2 <- as.data.frame(seg_km2$cluster)
km_res2$customer_id <- rownames(km_res2)
colnames(km_res2) <- c('cluster', 'customer_id')

rfm_customer$km2 <- km_res2[,'cluster']
head(rfm_customer)

aggregate(rfm_customer[,c(1:3)], list(rfm_customer$km2), mean)  # do not care to labels
#completely different Clusters!

table(rfm_customer$km2)

#visualize clusters
ggplot(data = rfm_customer, aes(x = freq, y = recency, color = factor(km2))) +
	geom_point() +
	ggtitle('kmeans - iter_2')  # too much difference between Clusters in km1 and km2 -> results are not robust!

# Model 2: CLARA
#First try
set.seed(1234)
seg_cl1 <- cluster::clara(rfm_customer_2, k = 5, samples = 10000, pamLike = T)

#Results
table(seg_cl1$cluster)  # 5 Clusters
cl_res1 <- as.data.frame(seg_cl1$cluster)  # cluster labels
cl_res1$customer_id <- rownames(cl_res1)
colnames(cl_res1) <- c('cluster', 'customer_id')

rfm_customer$cl1 <- cl_res1[,'cluster']
head(rfm_customer)

aggregate(rfm_customer[,c(1:3)], list(rfm_customer$cl1), mean)
table(rfm_customer$cl1)

#visualize clusters
ggplot(data = rfm_customer, aes(x = freq, y = recency, color = factor(cl1))) +
	geom_point() +
	ggtitle('clara - iter_1') +
	scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07", "#C3D7A4", "#52854C"))  # run_1 CLARA result -> similar to run_1 k-means result

#Second try
set.seed(12345678)
seg_cl2 <- cluster::clara(rfm_customer_2, k = 5, samples = 5000, pamLike = T)

#Results
table(seg_cl2$cluster)  # without care to labels: exactly same results with run_1 CLARA (exactly same population in each cluster)
table(seg_cl1$cluster)

cl_res2 <- as.data.frame(seg_cl2$cluster)  # cluster labels
cl_res2$customer_id <- rownames(cl_res2)
colnames(cl_res2) <- c('cluster', 'customer_id')

rfm_customer$cl2 <- cl_res2[,'cluster']
head(rfm_customer)

aggregate(rfm_customer[,c(1:3)], list(rfm_customer$cl2), mean)  # exact same output result
table(rfm_customer$cl2)

#visualize clusters
ggplot(data = rfm_customer, aes(x = freq, y = recency, color = factor(cl2))) +
	geom_point() +
	ggtitle('clara - iter_2') +
	scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07", "#C3D7A4", "#52854C"))  # there is no difference (changes) between results cl1 and cl2

#result: for this dataset, CLARA is more robust algorithm for Clustering compare to K-Means

# Model 3: Hierarchical K-Means
set.seed(1234)
seg_hk1 <- factoextra::hkmeans(rfm_customer_2, k = 5)

#Results
table(seg_hk1$cluster)  # again, different Clustering results -> all results are mathematically True (this is Clustering challenge)
hk_res1 <- as.data.frame(seg_hk1$cluster)
hk_res1$customer_id <- rownames(hk_res1)
colnames(hk_res1) <- c('cluster', 'customer_id')

rfm_customer$hk1 <- hk_res1[,'cluster']
head(rfm_customer)

aggregate(rfm_customer[,c(1:3)], list(rfm_customer$hk1), mean)
table(rfm_customer$hk1)

#visualize clusters
ggplot(data = rfm_customer, aes(x = freq, y = recency, color = factor(hk1)))+
	geom_point() +
	ggtitle("hkmeans - iter_1") + 
	scale_color_manual(values = c("#00AFBB", "#C3D7A4", "#E7B800", "#FC4E07", "#52854C"))


### Step 5: Model Evaluation ----
# Optimal number of clusters (CLARA):
#Elbow method: introduces an index for us to measure our Clustering quality and decide about number of Clusters
rfm_customer_2_sample <- rfm_customer_2[sample(1:nrow(rfm_customer_2), 5000), ]
plot_elbow_cl <- factoextra::fviz_nbclust(rfm_customer_2_sample, cluster::clara, method = 'wss')  # 10 times run clustering (once per different number of Clusters) then compare results based-on TWSS and choose the best k
plot_elbow_cl  # by increasing number of Clusters, `Total Within Sum of Squares` decreases

#which K is better? 5 is better
plot_elbow_cl <- plot_elbow_cl + 
		geom_vline(xintercept = 5, linetype = 2) +
		labs(subtitle = 'Elbow method_CLARA')  # Elbow breaks at 5
plot_elbow_cl
plot_elbow_cl$data  # TWSS values per different number of Clusters

#Silhouette method:
plot_silhouette_cl <- factoextra::fviz_nbclust(rfm_customer_2_sample, cluster::clara, method = 'silhouette') +
			labs(subtitle = 'Silhouette method_CLARA')
plot_silhouette_cl  # 2-cluster is better from statistical aspect based-on Silhouette method
plot_silhouette_cl$data

# Optimal number of clusters (Hierarchical K-Means):
#Elbow method:
plot_elbow_hk <- factoextra::fviz_nbclust(rfm_customer_2_sample, hkmeans, method = 'wss') +
			labs(subtitle = 'Elbow method_hkmeans')
plot_elbow_hk
plot_elbow_hk$data

#Silhouette method:
plot_silhouette_hk <- factoextra::fviz_nbclust(rfm_customer_2_sample, hkmeans, method = 'silhouette') +
			labs(subtitle = 'Silhouette method_hkmeans')
plot_silhouette_hk
plot_silhouette_hk$data

#now, we can compare models based on their silhouette (max) and twss (min) index -> based-on statistical indexes, hklust results is better than others
# but there is still a question: are these Clusters good from Business aspect? which Clustering results are better? -> we can not answer this question in ML area, we can answer this question in applied area
