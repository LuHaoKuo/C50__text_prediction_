library(tm) 
library(magrittr)
library(slam)
library(proxy)
library(glmnet)

readerPlain = function(fname){
  readPlain(elem=list(content=readLines(fname)), 
            id=fname, language='en') }
setwd("/Users/k.vincent/STA380/data/ReutersC50/C50train")
doc_list = Sys.glob('*')
file_list = Sys.glob(paste0(doc_list, '/*.txt'))
#paste0('../data/ReutersC50/C50train',"/auther","/*")
#i = 1

file_list = Sys.glob(paste0(doc_list, '/*.txt'))
temp = lapply(file_list, readerPlain) 


mynames = file_list %>%
{ strsplit(., '/', fixed=TRUE) } %>%
{ lapply(., tail, n=2) } %>%
{ lapply(., paste0, collapse = '') } %>%
  unlist
#mynames[[50]] = mynames[[51]]
#mynames=mynames[1:50]
names(temp) = mynames


documents_raw = VCorpus(VectorSource(temp))

my_documents = documents_raw
my_documents = tm_map(my_documents, content_transformer(tolower)) # make everything lowercase
my_documents = tm_map(my_documents, content_transformer(removeNumbers)) # remove numbers
my_documents = tm_map(my_documents, content_transformer(removePunctuation)) # remove punctuation
my_documents = tm_map(my_documents, content_transformer(stripWhitespace)) ## remove excess white-space

## Remove stopwords.  Always be careful with this: one person's trash is another one's treasure.
#stopwords("en")
#stopwords("SMART")
#my_documents = tm_map(my_documents, content_transformer(removeWords), stopwords("en"))


DTM = DocumentTermMatrix(my_documents)

DTM = removeSparseTerms(DTM, 0.95)
#findFreqTerms(DTM, 50)

# construct TF IDF weights
tfidf = weightTfIdf(DTM)


# Now PCA on tfidf
X = as.matrix(tfidf)
summary(colSums(X))
scrub_cols = which(colSums(X) == 0)
X = X[,-scrub_cols]

pca= prcomp(X, scale=TRUE)
plot(pca) 

X = pca$x[,1:100]

all  <- vector()
i = 1
for (auther in doc_list){
  all[i] = auther
  i = i+1
}
#ll[50] = all[51]
#all = all[1:50]

Y = vector()
for( i in 1:2500){
  Y[i] = all[ceiling(i/50)]
}

##test-set
readerPlain = function(fname){
  readPlain(elem=list(content=readLines(fname)), 
            id=fname, language='en') }
setwd("/Users/k.vincent/STA380/data/ReutersC50/C50test")
doc_list = Sys.glob('*')
file_list = Sys.glob(paste0(doc_list, '/*.txt'))


file_list = Sys.glob(paste0(doc_list, '/*.txt'))
cp2 = lapply(file_list, readerPlain) 


mynames = file_list %>%
{ strsplit(., '/', fixed=TRUE) } %>%
{ lapply(., tail, n=2) } %>%
{ lapply(., paste0, collapse = '') } %>%
  unlist
names(cp2) = mynames


documents_raw_1 = VCorpus(VectorSource(cp2))

my_documents1 = documents_raw_1
my_documents1 = tm_map(my_documents1, content_transformer(tolower)) # make everything lowercase
my_documents1 = tm_map(my_documents1, content_transformer(removeNumbers)) # remove numbers
my_documents1 = tm_map(my_documents1, content_transformer(removePunctuation)) # remove punctuation
my_documents1 = tm_map(my_documents1, content_transformer(stripWhitespace)) ## remove excess white-space

DTM_test = DocumentTermMatrix(my_documents1,control = list(dictionary=Terms(DTM)))
DTM_test = removeSparseTerms(DTM_test, 0.95)

tfidf_test = weightTfIdf(DTM_test)

X_test = as.matrix(tfidf_test)
scrub_cols = which(colSums(X_test) == 0)
X_test = X_test[,-scrub_cols]

####Column matching works:
train_pre_pc = as.matrix(tfidf)
scrub_cols = which(colSums(train_pre_pc) == 0)
train_pre_pc = train_pre_pc[,-scrub_cols]

train_name = colnames(train_pre_pc)
test_name = colnames(X_test)
sup = setdiff(train_name, test_name)

temp_x = data.frame(X_test)
for (colname_ in sup){
  temp_x[,colname_] = 0
}


##somehow there is still difference
#This can be identified using:
#setdiff(colnames(t), train_name)
#hereby I manually fix them
colnames(temp_x)[colnames(temp_x)=="for."] <- "for"
colnames(temp_x)[colnames(temp_x)=="next."] <- "next"
colnames(temp_x)[colnames(temp_x)=="while."] <- "while"
t = data.matrix(temp_x)
t <- t[, order(colnames(t))]
#####

#project the test set to the principal component spaces of the training set
test.data <- predict(pca, newdata =t)
test.data <- as.data.frame(test.data)
test.data <- test.data[,1:100]




out1 = glmnet(X, factor(Y), family="multinomial")
p1 = predict(out1, data.matrix(test.data), s=0.01, type = "response")

myPredict_for_out1 <- function(which_article){
  return(which.max(p1[which_article,,]))
}

Ya  <- vector()
i = 1
for (auther in doc_list){
  Ya[i] = auther
  i = i+1
}
#Ya
real = vector()
for( i in 1:2500){
  real[i] = Ya[ceiling(i/50)]
}


aut = vector()
for (i in 1:2500){
  aut[i] =names(myPredict_for_out1(i))
}

library(MLmetrics)
Accuracy(aut, real)

library(caret)
(table(aut,real)%>%confusionMatrix)$byClass[,"Balanced Accuracy"]
#RF
library(randomForest)

fY = factor(Y)
dfX =data.frame(X)
XY = cbind(dfX, fY)

rffit = randomForest(fY~.,data=XY,ntree=500)
prf<- predict(rffit, newdata = test.data)
Accuracy(prf, factor(real))

library(caret)
(table(prf,real)%>%confusionMatrix)$byClass[,"Balanced Accuracy"]






##Which articles are the most difficult to identify?
cosine_docs = function(dtm) {
  crossprod_simple_triplet_matrix(t(dtm))/(sqrt(col_sums(t(dtm)^2) %*% t(col_sums(t(dtm)^2))))
}

# use the function to compute pairwise cosine similarity for all documents
cosine_mat = cosine_docs(tfidf)

myStore = data.frame()
for(i in 1:2500){
  myStore[i,1] = as.numeric(i)
  myStore[i,2] = as.numeric(sort(cosine_mat[i,], decreasing=F)[1]%>%names)
  myStore[i,3] = sort(cosine_mat[i,], decreasing=F)[1]
}
colnames(myStore)<- c("Article_1", "Article_2", "Cosine_Distance")

#These are the articles who are very similar to each other
myrank = myStore[order(myStore$Cosine_Distance),]
#drop reversed duplicates
temp1 = apply(myrank[,1:2],1,function(x) paste(sort(x),collapse=''))
myrank[!duplicated(gsub(" ", "", temp1, fixed = TRUE)),]

#These are the corresponding authers
myrank1 = myrank[!duplicated(gsub(" ", "", temp1, fixed = TRUE)),]
myrank1$Article_1 = ceiling(myrank1$Article_1/50)
myrank1$Article_2 = ceiling(myrank1$Article_2/50)
myrank1 = myrank1[order(myrank1$Cosine_Distance),]


#pragma only after myrank1 defined
myThreshold<- function(threshold){
  local_df = myrank1[myrank1[,3]<threshold,]
  tr = apply(local_df[,1:2],1,function(x) paste(sort(x),collapse='-'))%>%table
  return(tr[order(tr, decreasing = TRUE)])
}

myThreshold(0.001)%>%head


