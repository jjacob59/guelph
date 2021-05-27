library(hrbrthemes)
library(viridis)
library(forcats)
# BiocManager::install("ComplexHeatmap")
library(ComplexHeatmap)
library(circlize)
library(reticulate); library(tidyverse); library(ggpubr)
library(cvms)
library(broom)    # tidy()
library(rlist); library(feather); library(imager)

set.seed(1)


grades <- c("white", "opaque", "gray", "shmoo", "pseudohyphae", "hyphae")
grade_paths <-  c("white", "white-opaque", "white-opaque-gray", "white-opaque-gray-shmoo", 
                  "white-opaque-gray-shmoo-pseudohyphae", "white-opaque-gray-shmoo-pseudohyphae-hyphae")
grades_abbr <- c("w", "o", "g", "s", "p", "h")
names(grades_abbr) <- grade_paths

short_filename <- function( fnames ) {
  f <- str_split( fnames, pattern = "/" )
  return( unlist( lapply( f , "[[", length(f[[1]]) )) )  }


make_unique_by_iou <- function( hallucin, upper_bound ){
  
  all_files <- unique(hallucin[["short_filename"]])
  final <- hallucin[-c(1:nrow(hallucin)),]
  
  for (i in 1:length(all_files)) {
    current_file <- all_files[i]
    hall <- hallucin %>% filter( short_filename == current_file)
    if (nrow(hall) < 2) { 
      final <- bind_rows(final, hall)
      next
    }
    
    ious <- matrix( nrow=nrow(hall), ncol = nrow(hall), data = 0)
    
    for (j in 1:(nrow(hall)-1)) {
      for (k in (j+1):nrow(hall)) {
        
        A <- c( hall[["bbox_1"]][j], hall[["bbox_2"]][j], hall[["bbox_3"]][j], hall[["bbox_4"]][j] )
        B <- c( hall[["bbox_1"]][k], hall[["bbox_2"]][k], hall[["bbox_3"]][k], hall[["bbox_4"]][k] )
        
        # x-dimension   
        xl <- max( A[1], B[1] )
        xr <- min( A[3], B[3] )
        if (xr <= xl) next
        
        yh <-min( A[2], B[2])
        yl <- max( A[4], B[4])
        if (yh >= yl) next
        
        num <- (xr - xl) * (yl - yh)
        denom <- num + ( (A[3]- A[1]) * (A[4]-A[2]) ) + ( (B[3]-B[1]) * (B[4]-B[4]) )
        
        ious[j, k] <-   num / denom
      } # end of k
    } # end of j
    
    to_remove <- c()
    while (max(ious) > upper_bound) {
      loc <- which(ious == max(ious), arr.ind = TRUE)
      to_remove <- c(to_remove, loc[1])
      ious[loc[1], ] <- 0
      ious[ , loc[1]] <- 0
    }
    
    if (length(to_remove) > 0) print( hall[to_remove, ] )
    
    ifelse(length(to_remove) > 0, final <- bind_rows(final, hall[-to_remove,]), final <- bind_rows(final, hall))
    
  }
  return(final)
}
