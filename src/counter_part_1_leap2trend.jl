using Word2Vec
using LinearAlgebra

function loadYears(fileName::String)::Vector{Int}
    file = open(fileName)
    years = Vector{Int}()
    for line in readlines(file)
        if line == ""
            continue
        end
        fields = split(line, ". ")
        try 
        year = parse(Int, fields[1])
        push!(years, year)
        catch
            println(line)
        end
    end
    close(file)
    years
end

function generateTemporalTrainFile(inFileName::String, outFileName::String, years::Vector{Int}, toYear::Int)
    fileIn = open(inFileName)
    fileOut = open(outFileName, "w")
    iLine = 1
    for line in readlines(fileIn)
        year = years[iLine]
        if year <= toYear
            println(fileOut, line)
        end
        iLine += 1
    end
    close(fileIn)
    close(fileOut)
end

function findHighFreqPhrases(fileName::String, freqThres::Int)::Vector{String}
    file = open(fileName)
    phraseFreqs = Dict{String, Int}()
    for line in readlines(file)
        phrases = split(line)
        for phrase in phrases
            phraseFreqs[phrase] = get(phraseFreqs, phrase, 0) + 1
        end
    end
    close(file)

    highFreqPhrases = Vector{String}()
    for (phrase, freq) in phraseFreqs
        if freq >= freqThres
            push!(highFreqPhrases, phrase)
        end
    end

    highFreqPhrases
end

function loadNormalizedVectors(fileName::String, highFreqPhrases::Vector{String})::Array{Float64, 2}
    wordVectors = wordvectors(fileName)
    numDims, vocabSize = size(wordVectors)
    numVecs = length(highFreqPhrases)
    vectors = Array{Float64}(undef, (numDims, numVecs))
    for (id, phrase) in enumerate(highFreqPhrases)
        if in_vocabulary(wordVectors, phrase)
            vector = get_vector(wordVectors, phrase)
            w = 1.0 / norm(vector)
            vectors[:,i] = w .* vector
        else
            vectors[:,i] = 0.0
        end
    end
    vectors
end

years = loadYears("DATR/data/NIPS.txt")
for toYear in 2010:2020
    generateTemporalTrainFile("DATR/workspace/NIPS.w2v.training.txt", "DATR/workspace/NIPS.w2v.$toYear.txt", years, toYear)
    word2vec("DATR/workspace/NIPS.w2v.$toYear.txt", "DATR/workspace/NIPS.w2v.model-$toYear.txt", size=200, iter=100, min_count=3, verbose=true)
end

highFreqPhrases = findHighFreqPhrases("DATR/workspace/NIPS.w2v.2015.txt", 5)

prevRankMat = nothing
for toYear in [2010, 2020]
    vectors = loadNormalizedVectors("DATR/workspace/NIPS.w2v.$toYear.txt", highFreqPhrases)
    simMat = 0.5 .+ 0.5 .* (vectors' * vectors)
    simVec = simMat[:]
    simRank = sortperm(simVec, rev=true)
    rankMat = Array{Int,2}(undef, size(simMat))
    for (i, k) in enumerate(simRank)
        rankMat[k] = i
    end
    if prevRankMat !== nothing
        deltaRankMat = rankMat - prevRankMat
        deltaRankVec = deltaRankMat[:]
        jumpRank = sortperm(deltaRankVec, rev=true)
        saveTopJumps
    end
    prevRankMat = rankMat
end