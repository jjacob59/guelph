if (tensorflow::tf$executing_eagerly())
  tensorflow::tf$compat$v1$disable_eager_execution()


library(keras)
library(bmp)
library(imager)
library(tidyverse)
library(abind)
library(ggpubr)

K <- keras::backend()

exp = "exp27"
numba = "_exp27_"
thresh = 0.25

# Paths ----------------------


train_events <- file.path( CANDESCENCE, "vaes", paste0("train_object__",  exp))
val_events <- file.path( CANDESCENCE, "vaes", paste0("val_object__",  exp))
test_events <- file.path( CANDESCENCE, "vaes", paste0("test_object__",  exp, "_test__thresh_", thresh))
  

keras_model_dir <-  file.path(CANDESCENCE, "vaes", "keras_models", exp)
output <- file.path(CANDESCENCE, "vaes", exp)

classes <- c("Yeast White" ,   "Budding White",  "Yeast Opaque",   "Budding Opaque",
             "Yeast Gray",     "Budding Gray",  
             "Shmoo", "Artifact",       "Unknown",        
                      "Pseudohyphae",   "Hyphae",        
             "P-junction",      "H-junction",
             "P-Start",        "H-Start"         )


train_frac <- 1.0
val_frac <- 1.0
test_frac <- 1.0

# input image dimensions
img_rows <- 128L
img_cols <- 128L
# color channels (1 = grayscale, 3 = RGB)
img_chns <- 1L

# Data preparation --------------------------------------------------------


image_tensor <- function( targets ) {
  tmp <-  lapply(targets, FUN = function(t) {
    tmp <- load.image(t) %>% as.array
    return( array_reshape(tmp[,,1,1], c(128, 128, 1), order = "F" ) )
  }) 
  return( abind( tmp, along = 0 ) )
}

get_labels <- function( fs ) {
  fs_1 <- str_split(fs, pattern = "/")
  fs_2 <-  lapply(fs_1, "[[", 8)
  fs_3 <- lapply(str_split( fs_2, pattern=".bmp"), "[[", 1)
  fs_4 <- as.numeric( unlist( lapply(str_split(fs_3, pattern = "_"), "[[", 8) ) )
  fs_5 <- unlist(lapply( fs_4, FUN = function(f) return( classes[f+1] )))
  return( fs_5 )
} 

get_test_labels <- function( fs ) {
  fs_1 <- str_split(fs, pattern = "/")
  fs_2 <-  lapply(fs_1, "[[", 8)
  fs_3 <- lapply(str_split( fs_2, pattern=".bmp"), "[[", 1)
  fs_4 <- unlist( lapply(str_split(fs_3, pattern = "_"), "[[", 2) ) 
  tmp <- which(fs_4 == "Unknown ")
  fs_4[tmp ] <- "Unknown"
  return( fs_4 )
} 

wgo <- c("Yeast White" ,   "Budding White",  "Yeast Opaque",   "Budding Opaque",
         "Yeast Gray",     "Budding Gray",  
         "Artifact",         
         "Shmoo",                 
         "H-Start",        "P-Start"  )



files_x_train <- list.files(train_events, full.names = TRUE)
labels_x_train <- get_labels(files_x_train)
keep <- which(labels_x_train %in% wgo)
files_x_train <- files_x_train[keep]
labels_x_train <- labels_x_train[keep]

files_x_val <- list.files(val_events, full.names = TRUE)
labels_x_val <- get_labels(files_x_val)
keep <- which(labels_x_val %in% wgo)
files_x_val <- files_x_val[keep]
labels_x_val <- labels_x_val[keep]

# test_fs <- list.files(test_events, full.names = TRUE)
# files_x_test <- sample(test_fs, length(test_fs) * test_frac, replace = FALSE)
# labels_x_test <- get_test_labels(files_x_test)
# 
# keep <- which(labels_x_test %in% wgo)
# files_x_test <- files_x_test[keep]
# labels_x_test <- labels_x_test[keep]



x_train <- image_tensor( files_x_train ) 
x_val  <- image_tensor( files_x_val )
# x_test <- image_tensor( files_x_test )

#cat("\n Dimensions of train: ", dim(x_train), 
#    "\t Dimensions of val: ", dim(x_val), 
#    "\t Dimensions of test: ", dim(x_test), "\n")

#### Parameterization ####

set.seed(1)


# number of convolutional filters to use
filters <- 64L

# convolution kernel size
num_conv <- 5L

latent_dim <- 2L
intermediate_dim <-  32L
epsilon_std <- 1.0

# training parameters
batch_size <- 100L
eps <- 10L


#### Model Construction ####

original_img_size <- c(img_rows, img_cols, img_chns)

x <- layer_input(shape = c(original_img_size))

conv_1 <- layer_conv_2d(  #conv2d_20
  x,
  filters = img_chns,
  kernel_size = c(2L, 2L),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

conv_2 <- layer_conv_2d(  #convd_21
  conv_1,
  filters = filters,
  kernel_size = c(2L, 2L),
  strides = c(2L, 2L),
  padding = "same",
  activation = "relu"
)

conv_3 <- layer_conv_2d( #convd_22
  conv_2,
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

conv_4 <- layer_conv_2d( #convd_23
  conv_3,
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

flat <- layer_flatten(conv_4)
hidden <- layer_dense(flat, units = intermediate_dim, activation = "relu")

z_mean <- layer_dense(hidden, units = latent_dim)
z_log_var <- layer_dense(hidden, units = latent_dim)

sampling <- function(args) {
  z_mean <- args[, 1:(latent_dim)]
  z_log_var <- args[, (latent_dim + 1):(2 * latent_dim)]
  
  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]),
    mean = 0.,
    stddev = epsilon_std
  )
  z_mean + k_exp(z_log_var) * epsilon
}

z <- layer_concatenate(list(z_mean, z_log_var)) %>% layer_lambda(sampling)

#output_shape <- c(batch_size, 14L, 14L, filters)
output_shape <- c(batch_size, 64L, 64L, filters)
#output_shape <- c(batch_size, 128L, 128L, filters)


decoder_hidden <- layer_dense(units = intermediate_dim, activation = "relu")
decoder_upsample <- layer_dense(units = prod(output_shape[-1]), activation = "relu")

decoder_reshape <- layer_reshape(target_shape = output_shape[-1])
decoder_deconv_1 <- layer_conv_2d_transpose(
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

decoder_deconv_2 <- layer_conv_2d_transpose(
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

decoder_deconv_3_upsample <- layer_conv_2d_transpose(
  filters = filters,
  kernel_size = c(3L, 3L),
  strides = c(2L, 2L),
  padding = "valid",
  activation = "relu"
)

decoder_mean_squash <- layer_conv_2d(
  filters = img_chns,
  kernel_size = c(2L, 2L),
  strides = c(1L, 1L),
  padding = "valid",
  activation = "sigmoid"
)

hidden_decoded <- decoder_hidden(z)
up_decoded <- decoder_upsample(hidden_decoded)
reshape_decoded <- decoder_reshape(up_decoded)
deconv_1_decoded <- decoder_deconv_1(reshape_decoded)
deconv_2_decoded <- decoder_deconv_2(deconv_1_decoded)
x_decoded_relu <- decoder_deconv_3_upsample(deconv_2_decoded)
x_decoded_mean_squash <- decoder_mean_squash(x_decoded_relu)

# custom loss function
vae_loss <- function(x, x_decoded_mean_squash) {
  beta = 1.0
  x <- k_flatten(x)
  x_decoded_mean_squash <- k_flatten(x_decoded_mean_squash)
  xent_loss <- 1.0 * img_rows * img_cols *
    loss_binary_crossentropy(x, x_decoded_mean_squash)
  kl_loss <- -0.5 * k_mean(1 + z_log_var - k_square(z_mean) -
                             beta* k_exp(z_log_var), axis = -1L)
  k_mean(xent_loss + kl_loss)
}

## variational autoencoder
vae <- keras_model(x, x_decoded_mean_squash)
#vae %>% compile(optimizer = "rmsprop", loss = vae_loss)
vae %>% compile(optimizer = optimizer_adam(), loss = vae_loss)  # better than rmsprop
#vae %>% compile(optimizer = optimizer_nadam(), loss = vae_loss)  # not bad. a bit compressed. compare with adam
#vae %>% compile(optimizer = optimizer_adagrad(), loss = vae_loss)  # also in contention. A bit compressed
#vae %>% compile(optimizer = optimizer_adadelta(), loss = vae_loss) # also good


optimiza <- "adam"

summary(vae)

## encoder: model to project inputs on the latent space
encoder <- keras_model(x, z_mean)

## build a digit generator that can sample from the learned distribution
gen_decoder_input <- layer_input(shape = latent_dim)
gen_hidden_decoded <- decoder_hidden(gen_decoder_input)
gen_up_decoded <- decoder_upsample(gen_hidden_decoded)
gen_reshape_decoded <- decoder_reshape(gen_up_decoded)
gen_deconv_1_decoded <- decoder_deconv_1(gen_reshape_decoded)
gen_deconv_2_decoded <- decoder_deconv_2(gen_deconv_1_decoded)
gen_x_decoded_relu <- decoder_deconv_3_upsample(gen_deconv_2_decoded)
gen_x_decoded_mean_squash <- decoder_mean_squash(gen_x_decoded_relu)
generator <- keras_model(gen_decoder_input, gen_x_decoded_mean_squash)



#### Model Fitting ####


with(tensorflow::tf$device('GPU:9'), {
  vae %>% fit(
    x_train, x_train, 
    shuffle = TRUE, 
    epochs = eps, 
    batch_size = batch_size, 
    validation_data = list(x_val, x_val)
  )
})

gc()

#vae %>% model.save((file.path(save_keras_dir, "first_time"))
#vae %>% export_savedmodel(file.path(save_keras_dir, "first_time"), remove_learning_phase = FALSE)
vae %>% save_model_weights_tf(keras_model_dir)

                   

#### Visualizations ####

library(ggplot2)
library(dplyr)
library(pals)


# -> prep the test1 and test2 tibbles for umap

x_train_encoded <- predict(encoder, x_train, batch_size = batch_size)
x_val_encoded <- predict(encoder, x_val, batch_size = batch_size)
#x_test_encoded <- predict(encoder, x_test, batch_size = batch_size)


x_train_encoded <- x_train_encoded %>% as_tibble() %>% 
  mutate(class = as.factor(labels_x_train)) %>% mutate( filename = files_x_train)
x_train_encoded[['type']] <- "train"

x_val_encoded <- x_val_encoded %>% as_tibble() %>% 
  mutate(class = as.factor(labels_x_val))  %>% mutate( filename = files_x_val)
x_val_encoded[['type']] <- "val"

# x_test_encoded <- x_test_encoded %>% as_data_frame() %>%
#   mutate(class = as.factor(labels_x_test))
# x_test_encoded %>% 
#   ggplot(aes(x = V1, y = V2, colour = class)) + geom_point() +  scale_color_manual(values = watlington())
# x_test_encoded[['type']] <- "test"

x_all_encoded <- bind_rows( x_train_encoded, x_val_encoded )
#x_all_encoded <- bind_rows( x_all_encoded, x_test_encoded )
saveRDS(x_all_encoded, file = paste0("~/vae_all_", 2, ".Rdata"))


# <--------------- visualizations

# < ------ Panel A

# x <- umap( x_all_encoded %>% select( starts_with('V'))) 

xte <- x_all_encoded 
g1 <- xte %>% ggplot(aes(x = V1, y = V2, color = class) ) +
  geom_point(size = 0.8, alpha = 0.6) +
  scale_color_manual( values = 
                        c("black", "lightgreen", "lightblue", "pink", "purple", "orange", "yellow", "green", "blue", "red"))

#  scale_colour_manual(values = c("pink", "blue", "green"))
g1
ggsave(filename = "vae_final.png", plot = g1, dpi = 450)

art <- xte %>% filter(  class == "Yeast White")
art <- art %>% arrange( V1 )

hi <- tibble(V1 = NA, V2= NA, class = NA, filename = NA, type = NA) 
lo <- tibble(V1 = NA, V2= NA, class = NA, filename = NA, type = NA) 
for (i in seq(-6, 3,0.5)) {
  tmp <- art %>% filter( V1 > i-0.3 , V1 < i + 0.3) %>% arrange(V2)
  hi <- rbind(hi, tmp[ nrow(tmp), ])
  lo <- rbind(lo, tmp[1, ])
}

load.image(as.character(hi[14, 'filename'])) %>% plot 
load.image(as.character(lo[14, 'filename'])) %>% plot 

i <- 19
tmp <- load.image(as.character(hi[i, 'filename']))
imager::save.image(tmp, paste0("~/", "vae_sub_hi_", i, ".png") )
plot(tmp)

i <- 20
tmp <- load.image(as.character(lo[i, 'filename']))
imager::save.image(tmp, paste0("~/", "vae_sub_lo_", i, ".png") )
plot(tmp)

low_art <- xte %>% filter( V1 < -5, class == "Artifact")
mid1 <-  xte %>% filter( V1 > -4, V1 < -3.5, class == "Artifact")
mid2 <-  xte %>% filter( V1 > -2.5, V1 < -2.3, class == "Artifact")
mid3 <-  xte %>% filter( V1 > -1.5, V1 < -1, class == "Artifact")
mid4 <-  xte %>% filter( V1 > -0.25, V1 < 0.25, class == "Artifact")
mid4 <-  xte %>% filter( V1 > 1, V1 < 1.25, class == "Artifact")
high_art<- xte %>% filter( V1 > 2.5, class == "Artifact")

 load.image(as.character(low_art[1,'filename'])) %>% plot 
load.image(as.character(mid1[1,'filename'])) %>% plot 
 load.image(as.character(mid2[1,'filename'])) %>% plot 
 load.image(as.character(mid3[1,'filename'])) %>% plot 
 load.image(as.character(mid4[1,'filename'])) %>% plot 
 load.image(as.character(high_art[1, 'filename'])) %>% plot
