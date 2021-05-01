using Word2Vec
using LinearAlgebra
using MultivariateStats
using Plots

function loadNonStopWords(fileName::String)::Tuple{Vector{String},Dict{String,Int}}
    file = open(fileName)
    nonStopWords = Vector{String}()
    nonStopWordIDs = Dict{String,Int}()
    for phrase in readlines(file)
        if phrase != ""
            push!(nonStopWords, phrase)
            nonStopWordIDs[phrase] = length(nonStopWords)
        end
    end
    close(file)
    (nonStopWords, nonStopWordIDs)
end

function normalizeVectors(wordVectors::WordVectors, nonStopWords::Vector{String})::Array{Float64,2}
    numDims, numWords = size(wordVectors)
    numVectors = length(nonStopWords)
    result = Array{Float64,2}(undef, (numDims, numVectors))
    for i in 1:numVectors
        vector = get_vector(wordVectors, nonStopWords[i])
        w = 1.0 / norm(vector)
        result[:,i] = vector * w
    end
    result
end

function loadPhraseIDGroups(fileName::String, nonStopWordIDs::Dict{String,Int})::Vector{Vector{Int}}
    file = open(fileName)
    phraseIDGroups = Vector{Vector{Int}}()
    for line in readlines(file)
        if length(line) == 0 
            continue
        end
        phrases = split(line)
        idGroup = Vector{Int}()
        for phrase in phrases
            if haskey(nonStopWordIDs, phrase)
                push!(idGroup, nonStopWordIDs[phrase])
            end
        end
        push!(phraseIDGroups, idGroup)
    end
    close(file)
    phraseIDGroups
end

function getCardinalities(nonStopWords::Vector{String}, nonStopWordIDs::Dict{String,Int}, phraseIDGroups::Vector{Vector{Int}})::Vector{Int}
    numPhrases = length(nonStopWords)
    idMap = Dict{Int,Int}()
    for (newID, phrase) in enumerate(nonStopWords)
        idMap[nonStopWordIDs[phrase]] = newID
    end

    phraseFreqs = fill(0, numPhrases)
    for idGroup in phraseIDGroups
        for id in idGroup
            phraseFreqs[idMap[id]] += 1
        end
    end

    phraseFreqs
end

function loadAspects(fileName::String, nonStopWordIDs::Dict{String,Int})::Tuple{Vector{Set{Int}},Vector{Int}}
    file = open(fileName)
    aspects = Vector{Set{Int}}()
    numPhrases = length(nonStopWordIDs)
    aspectIDs = Vector{Int}(undef, numPhrases)
    for line in readlines(file)
        fields = split(line)
        if length(fields) != 3
            continue
        end
        aspectID = parse(Int, fields[1])
        card = parse(Int, fields[2])
        phrase = fields[3]
        phraseID = nonStopWordIDs[phrase]
        if aspectID > length(aspects)
            @assert aspectID == length(aspects) + 1
            push!(aspects, Set{Int}())
        end
        push!(aspects[aspectID], phraseID)
        aspectIDs[phraseID] = aspectID
    end
    close(file)
    (aspects, aspectIDs)
end

function plotAspects(aspects::Vector{Set{Int}}, vectors::Array{Float64,2}, pca::PCA{Float64}, dim1::Integer, dim2::Integer)
    numAspects = length(aspects)
    numDims = size(vectors, 1)
    aspectVectors = Vector{Array{Float64,2}}(undef, numAspects)
    aspectFreqs = Vector{Vector{Int}}(undef, numAspects)
    for (aspectID, aspect) in enumerate(aspects)
        aspectSize = length(aspect)
        aspectPhraseIDs = [id for id in aspect]
        aspectVectors[aspectID] = Array{Float64,2}(undef, (numDims, aspectSize))
        for (localID, globalID) in enumerate(aspectPhraseIDs)
            aspectVectors[aspectID][:,localID] = vectors[:,globalID] 
        end
    end
    result1 = []
    result2 = []
    for aspectID in 1:numAspects
        aspectPCA = transform(pca, aspectVectors[aspectID])
        push!(result1, aspectPCA[dim1,:])
        push!(result2, aspectPCA[dim2,:])
        #p = scatter(aspectPCA[1,:],aspectPCA[2,:], marker=:circle, linewidth=0)
        #plot!(p,xlabel="PC1",ylabel="PC2")
    end
    (result1, result2)
end

wordVectors = wordvectors("DATR/workspace/NIPS.w2v.model.txt")
nonStopWords, nonStopWordIDs = loadNonStopWords("DATR/workspace/NIPS.nonStopWords.txt")
vectors = normalizeVectors(wordVectors, nonStopWords)
phraseIDGroups = loadPhraseIDGroups("DATR/workspace/NIPS.w2v.training.txt", nonStopWordIDs)
cardinalities = getCardinalities(nonStopWords, nonStopWordIDs, phraseIDGroups)
aspects1, aspectIDs1 = loadAspects("DATR/workspace/NIPS.aspects.1.txt", nonStopWordIDs)
aspects2, aspectIDs2 = loadAspects("DATR/workspace/NIPS.aspects.2.txt", nonStopWordIDs)
aspects3, aspectIDs3 = loadAspects("DATR/workspace/NIPS.aspects.3.txt", nonStopWordIDs)
pca = fit(PCA, vectors)
X, Y = plotAspects(aspects1, vectors, pca, 1, 2)
scatter(X,Y)
X, Y = plotAspects(aspects2, vectors, pca, 1, 2)
scatter(X,Y)
X, Y = plotAspects(aspects3, vectors, pca, 1, 2)
scatter(X,Y)

X, Y = plotAspects(aspects1, vectors, pca, 1, 3)
scatter(X,Y)
X, Y = plotAspects(aspects2, vectors, pca, 1, 3)
scatter(X,Y)
X, Y = plotAspects(aspects3, vectors, pca, 1, 3)
scatter(X,Y)

X, Y = plotAspects(aspects1, vectors, pca, 2, 3)
scatter(X,Y)
X, Y = plotAspects(aspects2, vectors, pca, 2, 3)
scatter(X,Y)
X, Y = plotAspects(aspects3, vectors, pca, 2, 3)
scatter(X,Y)