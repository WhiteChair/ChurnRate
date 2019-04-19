setwd("C:\\Users\\danie\\Desktop\\spark\\StreamingData\\Unstructured")

files <- list.files(full.names = T)
for (i in 1:length(files)) {
  subfiles <- list.files(files[i])  
  if (length(subfiles) == 1) {
    unlink(files[i], recursive = T)
  }
}
