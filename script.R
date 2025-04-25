library(ggplot2)
library(cluster)
library(factoextra)

#################################################################################################
# 1.Loading Data
################################################################################################


ratings <- read.table("data/ml-100k/u.data", sep="\t", 
                      col.names=c("user_id", "item_id", "rating", "timestamp"))

users <- read.table("data/ml-100k/u.user", sep="|", 
                    col.names=c("user_id", "age", "gender", "occupation", "zip_code"))

movies <- read.table("data/ml-100k/u.item", sep="|", 
                     col.names=c("item_id", "movie_title", "release_date", "video_release_date", 
                                 "IMDb_URL", "unknown", "Action", "Adventure", "Animation", "Children", 
                                 "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film_Noir", 
                                 "Horror", "Musical", "Mystery", "Romance", "Sci_Fi", "Thriller", 
                                 "War", "Western"),
                     fill = TRUE, encoding = "latin1", quote="")



merged_df <- merge(ratings, users, by = "user_id", all.x = TRUE)
merged_df <- merge(merged_df, movies, by = "item_id", all.x = TRUE)


str(merged_df)

head(merged_df)

data_types <- sapply(merged_df, class)
data_type_count <- table(data_types)
print(data_type_count)

################################################################################
# 2 - Data Transformation and Cleaning
################################################################################

# Select only numeric columns
numeric_features <- merged_df[, sapply(merged_df, is.numeric)]

# Remove the non-numeric columns manually
numeric_features <- merged_df[, !colnames(merged_df) %in% c("user_id", "item_id", "occupation", 
                                                            "movie_title", "release_date", 
                                                            "timestamp", "zip_code", 
                                                            "IMDb_URL", "video_release_date")]

## 2.1 Convert categorical variables into numeric format.

numeric_features$gender <- ifelse(numeric_features$gender == "M", 1, 0)

# Reduce Dataset Size
set.seed(42)
sample_size <- 20000  # Reduce data size
df_final <- numeric_features[sample(1:nrow(numeric_features), sample_size), ]

str(df_final)

data_types <- sapply(df_final, class)
data_type_count <- table(data_types)
print(data_type_count)

###############################################################################
# 3 PCA Analysis 
##############################################################################

scaled_features <- scale(df_final)  # Standardize features

pca_result <- prcomp(scaled_features, center = TRUE, scale. = TRUE)

summary(pca_result)  # View variance explained

# Plot the first two principal components to visualize the results
biplot(pca_result, scale = 0)

# Load necessary library
library(rgl)

# Perform PCA (assuming pca_result is already computed)
pca_data <- data.frame(pca_result$x[, 1:3])  # Extract first 3 PCs
pca_data$rating <- as.factor(df_final$rating)  # Convert rating to factor for coloring

# Define colors based on rating
rating_levels <- sort(unique(df_final$rating))  # Ensure sorted order
rating_colors <- rainbow(length(rating_levels))  # Generate distinct colors
colors <- rating_colors[as.numeric(pca_data$rating)]  # Assign colors to points

# Open an interactive 3D plot with rgl
open3d()
plot3d(pca_data$PC1, pca_data$PC2, pca_data$PC3, col = colors, size = 0.5, 
       xlab = "PC1", ylab = "PC2", zlab = "PC3", 
       main = "PCA: Rating Representation in PC1, PC2, PC3")


legend3d("topright", legend = rating_levels, pch = 19, col = rating_colors, title = "Ratings")

# Define PCA Components

# Get the eigenvalues (sdev squared)
eigenvalues <- (pca_result$sdev)^2

# Apply the Kaiser Criterion: Retain components with eigenvalue > 1
components_to_retain <- which(eigenvalues > 1)

# Number of components to retain
num_components <- length(components_to_retain)

# Print the components to retain and their eigenvalues
cat("Number of components to retain:", num_components, "\n")

k<- num_components # Critério de Kaiser
pca_reduced <- as.data.frame(pca_result$x[, 1:k])  # Keep first 'k' principal components

# Check dimensions
dim(pca_reduced)  # Should be (num_samples, k)


# Compute the proportion of variance explained by each principal component
variance_explained <- pca_result$sdev^2 / sum(pca_result$sdev^2)

# Compute the cumulative variance
cumulative_variance <- cumsum(variance_explained)

# Find the number of components that explain at least 90% of the variance
num_components_90 <- which(cumulative_variance >= 0.90)[1]

# Print results
cat("Number of components needed to explain at least 90% variance:", num_components_90, "\n")
#######################################################################################################


## 4.1 K-Means Clustering

#####################################################################################################

library(ggplot2)

# Use the Elbow Method to find the optimal number of clusters
wss <- numeric(20)  # Store within-cluster sum of squares for 1 to 10 clusters
for (i in 1:20) {
  kmeans_temp <- kmeans(pca_reduced, centers = i, nstart = 25)
  wss[i] <- kmeans_temp$tot.withinss
}

# Create a data frame to use for ggplot
elbow_data <- data.frame(Clusters = 1:20, WSS = wss)

# Create the Elbow plot using ggplot2 with integer x-axis labels
ggplot(elbow_data, aes(x = Clusters, y = WSS)) +
  geom_line() + 
  geom_point() +
  ggtitle("Elbow Method for Optimal Number of Clusters") +
  xlab("Number of Clusters") + 
  ylab("Total Within-Cluster Sum of Squares") +
  scale_x_continuous(breaks = 1:20) +  # Set x-axis labels as integers
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))


set.seed(123)  # For reproducibility
k_optimal <- 9  

kmeans_model <- kmeans(pca_reduced, centers = k_optimal, nstart = 25)

# Add cluster labels to the dataset
pca_reduced$cluster <- as.factor(kmeans_model$cluster)

# View cluster distribution
table(pca_reduced$cluster)

# Plot clusters in 3D using rgl
open3d()

# Define colors based on cluster labels
colors <- rainbow(k_optimal)[as.numeric(pca_reduced$cluster)]

# Plot the 3D scatter plot
plot3d(pca_reduced$PC1, pca_reduced$PC2, pca_reduced$PC3, 
       col = colors, size = 2, 
       xlab = "PC1", ylab = "PC2", zlab = "PC3", 
       main = "3D PCA Plot with K-Means Clusters")
#type="s")

# Add legend for clusters
legend3d("topright", legend = levels(pca_reduced$cluster), 
         col = rainbow(k_optimal), pch = 19, 
         title = "Clusters", cex = 1)


cluster_profile <- aggregate(df_final, by = list(cluster = pca_reduced$cluster), FUN = mean)

knitr::kable(cluster_profile, caption = "Cluster Profile - Mean Values by Cluster") %>%
  kableExtra::kable_styling() %>%
  kableExtra::scroll_box(width = "100%", height = "400px")

### Cluster Validation

sil_score_kmeans <- silhouette(kmeans_model$cluster, dist(pca_reduced))

cat("Average Silhouette Score:", mean(sil_score_kmeans[, 3]))

library(ggplot2)
library(cluster)

# Convert silhouette object to a data frame
sil_df <- data.frame(
  cluster = as.factor(sil_score_kmeans[, 1]),
  width = sil_score_kmeans[, 3]
)

# Plot with ggplot2
ggplot(sil_df, aes(x = cluster, y = width, fill = cluster)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Silhouette Widths for K-Means Clustering",
       x = "Cluster",
       y = "Silhouette Width")

###########################################################################
## 4.2 Hierarchical Clustering

###########################################################################

# Function to compute and plot average silhouette score

silhouette_analysis <- function(data, distance_metrics = c("euclidean", "manhattan","cosine"),
                                linkages = c("single","complete", "average", "ward.D"), k_values = 5:9) {
  
  # Ensure data is numeric
  data_numeric <- as.matrix(data)
  
  results <- data.frame()  # Store silhouette results
  
  # Iterate over distance metrics
  for (dist_metric in distance_metrics) {
    
    # Compute distance matrix (cosine similarity for "cosine")
    if (dist_metric == "cosine") {
      distance_matrix <- proxy::dist(data_numeric, method = "cosine")
    } else {
      # Compute standard distance matrix for other metrics
      distance_matrix <- dist(data_numeric, method = dist_metric)
    }
    
    # Iterate over linkage methods
    for (linkage in linkages) {
      
      # Perform hierarchical clustering
      hc <- hclust(distance_matrix, method = linkage)
      
      # Iterate over k values (number of clusters)
      for (k in k_values) {
        
        # Cut the dendrogram to form clusters
        clusters <- cutree(hc, k = k)
        
        # Compute silhouette score
        sil <- silhouette(clusters, distance_matrix)
        avg_sil <- mean(sil[, 3])  # Extract average silhouette score
        
        # Store results
        results <- rbind(results, data.frame(
          Distance = dist_metric,
          Linkage = linkage,
          Clusters = k,
          Avg_Silhouette = avg_sil
        ))
      }
    }
  }
  
  # Plot the results
  ggplot(results, aes(x = Clusters, y = Avg_Silhouette, color = Linkage)) +
    geom_line() + geom_point(size = 2) +
    facet_wrap(~ Distance) +
    labs(title = "Average Silhouette Score for Different Linkage and Distance Metrics",
         x = "Number of Clusters (k)", y = "Average Silhouette Score") +
    theme_minimal()
}

silhouette_analysis(pca_reduced)


library(dendextend)  


library(proxy)
pca_numeric <- pca_reduced[, !names(pca_reduced) %in% "cluster"]

# Convert to a numeric matrix
pca_matrix <- as.matrix(pca_numeric)

cosine_sim_matrix <- proxy::dist(pca_matrix, method = "cosine")

# Perform hierarchical clustering using cosine similarity
hclust_result <- hclust(cosine_sim_matrix, method = "average")

# Plot the dendrogram
plot(hclust_result, main = "Dendrogram of Hierarchical Clustering", 
     xlab = "Observations", ylab = "Height", cex = 0.7)

# Optional: Color branches for better visualization
dend <- as.dendrogram(hclust_result)
dend <- color_branches(dend, k = 9)  
plot(dend, main = "Dendrogram")

# Cut tree into desired clusters (e.g., k = 5)
clusters <- cutree(hclust_result, k = 9)

# Add cluster labels to pca_reduced dataframe
pca_reduced$cluster <- as.factor(clusters)

# View cluster distribution
table(pca_reduced$cluster)

# Define distinct colors for clusters
cluster_levels <- levels(factor(pca_reduced$cluster))
cluster_colors <- rainbow(length(cluster_levels))  # Generates unique colors
names(cluster_colors) <- cluster_levels  # Map colors to cluster labels

# Assign colors temporarily
pca_reduced$color <- cluster_colors[as.character(pca_reduced$cluster)]  

# Set material properties to prevent shading effects
material3d(col = "black", shininess = 50, specular = "black")

# 3D Scatter Plot with Solid Spheres
plot3d(pca_reduced$PC1, pca_reduced$PC2, pca_reduced$PC3, 
       col = pca_reduced$color,  # Use colors for plotting
       size = 0.5, 
       xlab = "PC1", ylab = "PC2", zlab = "PC3", 
       main = "Hierarchical Clustering") 
#type = "s")  

# Add 3D legend with matching colors
legend3d("topright", legend = names(cluster_colors), 
         col = cluster_colors, pch = 16, # Use solid points
         title = "Clusters", cex = 1.2)

rglwidget()

# Remove the color column to keep dataset numeric
pca_reduced$color <- NULL

### Examine Cluster Profiles

# Aggregate original features by cluster labels
cluster_profile <- aggregate(df_final, by = list(cluster = pca_reduced$cluster), FUN = mean)

# Display cluster profile as a table
knitr::kable(cluster_profile, caption = "Cluster Profile - Mean Values by Cluster") %>%
  kableExtra::kable_styling() %>%
  kableExtra::scroll_box(width = "100%", height = "400px")

# Calculate the silhouette score

sil <- silhouette(clusters, cosine_sim_matrix)
avg_sil <- mean(sil[, 3])  # Extract average silhouette score

cat("Average Silhouette Score:", avg_sil)

# Convert silhouette object to a data frame
sil_df_hclust <- data.frame(
  cluster = as.factor(sil[, 1]),
  width = sil[, 3]
)

# Plot with ggplot2
ggplot(sil_df_hclust, aes(x = cluster, y = width, fill = cluster)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Silhouette Widths for Hierarchical Clustering",
       x = "Cluster",
       y = "Silhouette Width")


###############################################################################
## 4.3 DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

###############################################################################

library(dbscan)
library(proxy)
library(cluster)  # For silhouette function
library(ggplot2)   # For visualization

# Function to calculate silhouette scores for different eps and minPts in DBSCAN
silhouette_for_eps_minPts <- function(data, eps_values, minPts_values) {
  # Calculate the cosine similarity matrix
  cosine_sim_matrix <- proxy::dist(data, method = "cosine")
  
  # Initialize a data frame to store results
  silhouette_scores <- data.frame(eps = numeric(0), minPts = numeric(0), silhouette_score = numeric(0))
  
  # Iterate over each eps value and minPts value
  for (eps in eps_values) {
    for (minPts in minPts_values) {
      # Perform DBSCAN with the current eps and minPts
      dbscan_result <- dbscan(cosine_sim_matrix, eps = eps, minPts = minPts)
      
      # If DBSCAN found more than 1 cluster (ignoring noise, which is labeled -1)
      if (length(unique(dbscan_result$cluster)) > 1) {
        # Compute the silhouette score
        sil <- silhouette(dbscan_result$cluster, cosine_sim_matrix)
        avg_sil <- mean(sil[, 3], na.rm = TRUE)  # Extract the average silhouette score
        
        # Append to results
        silhouette_scores <- rbind(silhouette_scores, data.frame(eps = eps, minPts = minPts, silhouette_score = avg_sil))
      }
    }
  }
  
  # Return the silhouette scores data frame
  return(silhouette_scores)
}

# Define eps values and minPts values to test
eps_values <- seq(0.1, 0.4, by = 0.1)   # eps values from 0.1 to 0.4
minPts_values <- seq(9, 18, by = 3)     # minPts values from 9 to 18

pca_reduced <- pca_reduced[, !names(pca_reduced) %in% "cluster"]

# Calculate silhouette scores for each combination of eps and minPts
sil_scores <- silhouette_for_eps_minPts(pca_reduced, eps_values, minPts_values)

# Plot the silhouette scores for each eps and minPts
ggplot(sil_scores, aes(x = eps, y = silhouette_score, color = factor(minPts))) +
  geom_line() +
  geom_point() +
  labs(title = "Silhouette Score vs. eps and minPts (DBSCAN with Cosine Similarity)",
       x = "eps", y = "Average Silhouette Score", color = "minPts") +
  theme_minimal()



pca_reduced <- pca_reduced[, !names(pca_reduced) %in% "cluster"]

# Calculate cosine similarity matrix (using 'dist' function from 'proxy' package)
cosine_sim_matrix <- proxy::dist(pca_reduced, method = "cosine")

# Run DBSCAN with cosine similarity matrix
dbscan_result <- dbscan(cosine_sim_matrix, eps = 0.11, minPts = 18)

# Add cluster labels to data
pca_reduced$cluster <- as.factor(dbscan_result$cluster)  # "0" represents noise

# Check for NAs in the cluster column
table(pca_reduced$cluster)  # This will show if there are NA values in the cluster column



# Define distinct colors for clusters
cluster_levels <- levels(factor(pca_reduced$cluster))
cluster_colors <- rainbow(length(cluster_levels))  # Generates unique colors
names(cluster_colors) <- cluster_levels  # Map colors to cluster labels

# Assign colors temporarily
pca_reduced$color <- cluster_colors[as.character(pca_reduced$cluster)]  

# Set material properties to prevent shading effects
material3d(col = "black", shininess = 50, specular = "black")

# 3D Scatter Plot with Solid Spheres
plot3d(pca_reduced$PC1, pca_reduced$PC2, pca_reduced$PC3, 
       col = pca_reduced$color,  # Use colors for plotting
       size = 0.5, 
       xlab = "PC1", ylab = "PC2", zlab = "PC3", 
       main = "Hierarchical Clustering")
#type = "s")  

# Add 3D legend with matching colors
legend3d("topright", legend = names(cluster_colors), 
         col = cluster_colors, pch = 16, # Use solid points
         title = "Clusters", cex = 1.2)

# Convert cluster labels to numeric for silhouette calculation
# We will keep the factor labels for visualization but use numeric labels for calculations
pca_reduced$cluster_numeric <- as.numeric(as.character(pca_reduced$cluster))

# Add cluster labels to data
pca_reduced$cluster <- as.factor(dbscan_result$cluster)  # "-1" represents noise

# Compute the silhouette score
if (length(unique(dbscan_result$cluster)) > 1) {  # Only calculate if more than 1 cluster exists
  silhouette_score <- silhouette(dbscan_result$cluster, as.matrix(cosine_sim_matrix))
  
  # Get the average silhouette score
  avg_silhouette_score <- mean(silhouette_score[, 3], na.rm = TRUE)  # Remove NAs from noise
  
  # Print the average silhouette score
  print(paste("Average Silhouette Score:", avg_silhouette_score))
} else {
  print("DBSCAN found only one cluster or all points are noise. Silhouette score is not meaningful.")
}


# Convert silhouette object to a data frame for easier plotting
sil_df_dbscan <- data.frame(
  cluster = as.factor(silhouette_score[, 1]),  # Cluster labels
  width = silhouette_score[, 3]  # Silhouette width
)

# Plot silhouette scores for DBSCAN clusters
ggplot(sil_df_dbscan, aes(x = cluster, y = width, fill = cluster)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Silhouette Widths for DBSCAN Clustering",
       x = "Cluster",
       y = "Silhouette Width") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better readability

# Apply DBSCAN clustering
dbscan_result <- dbscan(cosine_sim_matrix , eps = .12, minPts = 25)

# Add cluster labels to data
pca_reduced$cluster <- as.factor(dbscan_result$cluster)  # "0" represents noise
table(pca_reduced$cluster) 

###############################################################################
## Gaussian Mixture Models (GMMs)
###############################################################################

library(mclust)
library(cluster)

# Initialize empty vectors to store ICL, Silhouette scores, and BIC values
icl_values <- numeric(9)
silhouette_scores <- numeric(9)
bic_values <- numeric(9)

# Loop through different values of G (number of clusters)
for (k in 2:10) {  # Starting from 2 clusters (avoid 1 cluster, as silhouette is not defined for it)
  # Fit the GMM model
  gmm_model <- Mclust(pca_reduced, G = k)
  
  # Store the ICL value
  icl_values[k] <- gmm_model$icl
  
  # Store the BIC value
  bic_values[k] <- gmm_model$bic
  
  # Compute the dissimilarity matrix using Euclidean distance
  dist_matrix <- dist(pca_reduced)  # Euclidean distance matrix
  
  # Calculate the Silhouette Score
  silhouette_score <- silhouette(gmm_model$classification, dist_matrix)  # Pass the correct dist_matrix
  
  # Extract the silhouette widths (the third column)
  if (inherits(silhouette_score, "silhouette")) {
    sil_width <- silhouette_score[, 3]  # The third column contains silhouette widths
    
    # Remove NA values and calculate the mean silhouette score
    sil_width <- sil_width[!is.na(sil_width)]
    
    if (length(sil_width) > 0) {
      silhouette_scores[k] <- mean(sil_width)
    } else {
      silhouette_scores[k] <- NA
    }
  } else {
    silhouette_scores[k] <- NA  # If silhouette object is invalid, set as NA
  }
}

# Create a data frame with the results
results <- data.frame(
  Clusters = 2:10,  # Adjusted the cluster range to match the loop
  ICL = icl_values[2:10],
  BIC = bic_values[2:10],
  Silhouette = silhouette_scores[2:10]
)

# Print the results
print(results)

# Find the best number of clusters based on ICL, BIC, and Silhouette
best_icl_clusters <- which.min(icl_values[2:10])  # Minimize ICL (adjusted for the range)
best_bic_clusters <- which.min(bic_values[2:10])  # Minimize BIC (adjusted for the range)
best_silhouette_clusters <- which.max(silhouette_scores[2:10])  # Maximize Silhouette Score

# Print the best number of clusters based on each metric
cat("Best number of clusters based on ICL:", best_icl_clusters, "\n")
cat("Best number of clusters based on BIC:", best_bic_clusters, "\n")
cat("Best number of clusters based on Silhouette:", best_silhouette_clusters, "\n")


par(mfrow = c(1, 2), mar = c(5, 4, 4, 2) + 0.1)  

# Plot BIC and ICL values vs number of clusters
plot(results$Clusters, results$BIC, type = "b", col = "blue", pch = 19, 
     xlab = "Number of Clusters", ylab = "BIC", main = "BIC vs Number of Clusters", 
     ylim = range(c(results$BIC, results$ICL)), cex.main = 1.2, cex.lab = 1.1)
lines(results$Clusters, results$ICL, type = "b", col = "red", pch = 19)
legend("topright", legend = c("BIC", "ICL"), col = c("blue", "red"), pch = 19, cex = 0.5)

# Plot silhouette scores vs number of clusters
plot(results$Clusters, results$Silhouette, type = "b", col = "green", pch = 19, 
     xlab = "Number of Clusters", ylab = "Silhouette Score", main = "Silhouette vs Number of Clusters", 
     cex.main = 1.2, cex.lab = 1.1)


optimal_G <- which.min(icl_values)
print(optimal_G)

# Load required libraries
library(mclust)  # For Gaussian Mixture Models
library(rgl)     # For 3D visualization


# Perform Gaussian Mixture Model (GMM) clustering using all 9 principal components
# Apply the GMM model with the optimal number of clusters
gmm_model <- Mclust(pca_reduced, G = 8)

pca_reduced$cluster <- as.factor(gmm_model$classification)  # Extract cluster labels

# Define distinct colors for clusters
cluster_levels <- levels(pca_reduced$cluster)  # Extract unique cluster labels
cluster_colors <- rainbow(length(cluster_levels))  # Generate distinct colors
names(cluster_colors) <- cluster_levels  # Map colors to cluster labels

# Assign colors temporarily for plotting
pca_reduced$color <- cluster_colors[as.character(pca_reduced$cluster)]

# Set material properties to prevent shading effects
material3d(col = "black", shininess = 50, specular = "black")

# 3D Scatter Plot (Using First 3 PCs for Visualization)
plot3d(pca_reduced$PC1, pca_reduced$PC2, pca_reduced$PC3, 
       col = pca_reduced$color,  # Use assigned colors
       size = 0.5, 
       xlab = "PC1", ylab = "PC2", zlab = "PC3", 
       main = "Gaussian Mixture Model Clustering") 
#type = "s")  

# Add 3D legend with correctly assigned colors
legend3d("topright", legend = names(cluster_colors), 
         col = cluster_colors, pch = 16, 
         title = "Clusters", cex = 1.2)

print(table(pca_reduced$cluster))  # Count points in each cluster

summary(gmm_model)  # Detailed model summary

# Assuming you already have silhouette scores
library(cluster)
library(ggplot2)

# Calculate silhouette score
silhouette_score <- silhouette(gmm_model$classification, dist(pca_reduced))

# Convert silhouette object to a data frame
silhouette_df <- data.frame(
  cluster = as.factor(silhouette_score[, 1]),  # Extract cluster labels
  silhouette_width = silhouette_score[, 3]    # Extract silhouette widths
)

# Create a boxplot for silhouette scores by cluster
ggplot(silhouette_df, aes(x = cluster, y = silhouette_width)) +
  geom_boxplot(fill = "lightblue", color = "darkblue") +
  labs(title = "Boxplot of Silhouette Scores by Cluster", 
       x = "Cluster", 
       y = "Silhouette Width") +
  theme_minimal()

# Cluster size distribution
cluster_sizes <- table(gmm_model$classification)
barplot(cluster_sizes, main = "Cluster Size Distribution", col = "skyblue", ylab = "Number of Points", xlab = "Cluster")


library(gridExtra)

# Create a density plot for each Gaussian component
ggplot(pca_reduced, aes(x = PC1)) +
  geom_density(aes(y = after_stat(density), color = factor(gmm_model$classification)), 
               adjust = 1) +  # adjust bandwidth for smoother curve
  labs(title = "Density of GMM Components", x = "Principal Component 1", y = "Density") +
  theme_minimal()


# Get the means and covariance of the GMM
means <- gmm_model$parameters$mean
covariances <- gmm_model$parameters$variance$sigma

# Create a plot with data points and the ellipses
ggplot(pca_reduced, aes(x = PC1, y = PC2, color = factor(gmm_model$classification))) +
  geom_point() +
  stat_ellipse(data = pca_reduced, aes(x = PC1, y = PC2, color = factor(gmm_model$classification)),
               level = 0.95) +  # 95% ellipse level
  labs(title = "GMM Clusters with Ellipses", x = "Principal Component 1", y = "Principal Component 2") +
  theme_minimal()

print(gmm_model$G)

# Check the means (centroids) of the clusters
print(gmm_model$parameters$mean)

###############################################################################
##  Compare Clustering Results with Adjusted Rand Index (ARI):
###############################################################################
# Perform k-means clustering (e.g., 5 clusters)
kmeans_result <- kmeans(pca_reduced, centers = 9)
kmeans_clusters <- kmeans_result$cluster

# Perform hierarchical clustering
pca_numeric <- pca_reduced[, !names(pca_reduced) %in% "cluster"]

# Convert to a numeric matrix
pca_matrix <- as.matrix(pca_numeric)

cosine_sim_matrix <- proxy::dist(pca_matrix, method = "cosine")

# Perform hierarchical clustering using cosine similarity
hierarchical_result <- hclust(cosine_sim_matrix, method = "average")
# Cut the dendrogram into 5 clusters
hierarchical_clusters <- cutree(hierarchical_result, k = 9)

# Perform DBSCAN clustering
# Adjust eps and minPts to generate 5 clusters
library(dbscan)
dbscan_result <- dbscan(cosine_sim_matrix , eps = 0.12, minPts = 18)


# Add cluster labels to data
pca_reduced$cluster <- as.factor(dbscan_result$cluster) 
dbscan_clusters <- ifelse(dbscan_result$cluster == 0, 0, dbscan_result$cluster)

# Perform GMM clustering 
library(mclust)
gmm_result <- Mclust(pca_reduced, G = 8)  
gmm_clusters <- gmm_result$classification

# Function to compute Fowlkes-Mallows Index (FMI)
fowlkes_mallows <- function(cluster1, cluster2) {
  # Create a contingency table of the clusters
  table_ <- table(cluster1, cluster2)
  
  # Compute the number of pairs
  tp <- sum(choose(table_, 2))
  fp_fn <- sum(choose(table_ - diag(table_), 2))
  
  # Fowlkes-Mallows Index formula
  FMI <- tp / sqrt((tp + fp_fn) * (tp + fp_fn))
  return(FMI)
}

# Compute FMI for all combinations
fmi_kmeans_hierarchical <- fowlkes_mallows(kmeans_clusters, hierarchical_clusters)
fmi_kmeans_dbscan <- fowlkes_mallows(kmeans_clusters, dbscan_clusters)
fmi_kmeans_gmm <- fowlkes_mallows(kmeans_clusters, gmm_clusters)
fmi_dbscan_gmm <- fowlkes_mallows(dbscan_clusters, gmm_clusters)
fmi_hierarchical_dbscan <- fowlkes_mallows(hierarchical_clusters, dbscan_clusters)
fmi_hierarchical_gmm <- fowlkes_mallows(hierarchical_clusters, gmm_clusters)

# Store results
fmi_results <- data.frame(
  Method1 = c("k-means", "k-means", "k-means", "DBSCAN", "DBSCAN", "hierarchical"),
  Method2 = c("hierarchical", "DBSCAN", "GMM", "hierarchical", "GMM", "GMM"),
  FMI = c(fmi_kmeans_hierarchical, fmi_kmeans_dbscan, fmi_kmeans_gmm,
          fmi_dbscan_gmm, fmi_hierarchical_dbscan, fmi_hierarchical_gmm)
)

# Print results
print(fmi_results)

# Load necessary libraries

library(reshape2)

# Create a dataframe with FMI results
fmi_results <- data.frame(
  Method1 = c("k-means", "k-means", "k-means", "DBSCAN", "DBSCAN", "hierarchical"),
  Method2 = c("hierarchical", "DBSCAN", "GMM", "hierarchical", "GMM", "GMM"),
  FMI = c(0.5049704, 0.3694232, 0.3559871, 0.3291647, 0.2027890, 0.4408476)
)

# Create a heatmap-friendly matrix
fmi_matrix <- acast(fmi_results, Method1 ~ Method2, value.var = "FMI")

# Convert to long format for ggplot2
fmi_long <- melt(fmi_matrix, na.rm = TRUE)

# Plot the heatmap
ggplot(fmi_long, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  geom_text(aes(label = round(value, 3)), color = "white", size = 5) +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_minimal() +
  labs(title = "Fowlkes-Mallows Index (FMI) Heatmap",
       x = "Method 1",
       y = "Method 2",
       fill = "FMI Score")



