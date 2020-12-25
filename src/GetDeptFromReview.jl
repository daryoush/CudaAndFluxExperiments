
## ---
using CSV: read
using DataFrames
using Plots
using StatsBase
using DataStructures: SortedSet

## ---

df=read("data/Womens Clothing E-Commerce Reviews.csv", DataFrame)

#get 70% of the rows for training
trainingdataPercent=0.70
trainingRowIds=rand(1:nrow(df),convert(Int, round(nrow(df)/trainingdataPercent)))
validationRowIds=setdiff(1:nrow(df),trainingRowIds)

@assert isempty(intersect(trainingRowIds, validationRowIds))

trainingData=df[trainingRowIds,:]
validationData=df[validationRowIds, :]

## ---
names(trainginData)

#make sure both sets has all the dept names
traingDeptName=unique(trainginData."Department Name")
validDeptName=unique(validationData."Department Name")
@assert isempty(setdiff(traingDeptName, validDeptName))

#get an idea of the frequency of the review messages for the dept
countmap(trainginData."Department Name")

reviews=hcat(trainingData.Title, trainingData."Review Text")

labels=SortedSet()
for (t, r) in (trainingData.Title, trainingData."Review Text")
    push!(labels, (t*r)...)
end
labels
