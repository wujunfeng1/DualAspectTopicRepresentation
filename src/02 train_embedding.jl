using TextAnalysis
using Word2Vec
include("x1 io.jl")
include("x2 preprocessing.jl")
include("x3 aspect_aware_word_embedding.jl")

function mergeWithStrongLinks!(tokIDGroups::Vector{Vector{Int32}}, vocab::Dict{String,Int32}, strongLinkForest::Dict{IDPair,IDPostfixes})
    # insert strong-link phrases into vocab
    numStrongLinks = 0
    for (idPair, postfixes) in strongLinkForest
        numStrongLinks += length(postfixes) 
    end
    println("numStrongLinks: ", numStrongLinks)
    numOldPhrases = length(vocab)
    phrases = Vector{String}(undef, numOldPhrases + numStrongLinks)
    for (phrase, phraseID) in vocab
        phrases[phraseID] = phrase 
    end
    strongLinkIDs = Dict{Vector{Int32},Int32}()
    for (idPair, postfixes) in strongLinkForest
        prefix = phrases[idPair[1]] * "-" * phrases[idPair[2]]
        for postfix in postfixes
            phrase = prefix
            for id in postfix
                phrase *= "-" * phrases[id]
            end
            phraseID = length(vocab) + 1
            vocab[phrase] = phraseID
            phrases[phraseID] = phrase
            strongLinkIDs[[idPair...,postfix...]] = phraseID
        end
    end

    # scan through token groups to find occurence of strong-link phrases and convert them to the new entries of vocab
    numGroups = length(tokIDGroups)
    for (groupID, tokIDGroup) in enumerate(tokIDGroups)
        groupSize = length(tokIDGroup)
        strongLinksDetected = Dict{Int32,Int32}()
        for i in 1:groupSize - 1
            idPair = (tokIDGroup[i], tokIDGroup[i + 1])
            if haskey(strongLinkForest, idPair)
                postfixes = strongLinkForest[idPair]
                for postfix in postfixes
                    if i + 1 + length(postfix) > groupSize
                        continue
                    end
                    if tokIDGroup[i + 2:i + 1 + length(postfix)] == postfix
                        phraseSize = 2 + length(postfix)
                        if haskey(strongLinksDetected, i)
                            if strongLinksDetected[i] > phraseSize
                                continue
                            end
                        end
                        strongLinksDetected[i] = phraseSize
                    end
                end 
            end
        end
        phraseStarts = [i for i in 1:groupSize]
        for (phrasePos, phraseSize) in strongLinksDetected
            if phraseStarts[phrasePos] != phrasePos
                continue
            end
            if phraseStarts[phrasePos + phraseSize - 1] != phrasePos + phraseSize - 1
                continue
            end
            phraseStarts[phrasePos:phrasePos + phraseSize - 1] .= phrasePos
        end
        newTokIDGroup = Vector{Int32}()
        for i in 1:groupSize
            if phraseStarts[i] != i 
                continue
            end
            phraseEnd = i 
            for j in i + 1:groupSize
                if phraseStarts[j] != i
                    break
                end
                phraseEnd = j
            end
            if phraseEnd == i 
                push!(newTokIDGroup, tokIDGroup[i])
            else
                push!(newTokIDGroup, strongLinkIDs[tokIDGroup[i:phraseEnd]])
            end
        end
        tokIDGroups[groupID] = newTokIDGroup    
    end
end

function word2Phrase!(tokIDGroups::Vector{Vector{Int32}}, vocab::Dict{String,Int32}, maxIter::Int, minFreq::Int)
    # scan through tokIDGroups to build unigram freqs and bigram freqs
    numPhrases = length(vocab)
    phrases = Vector{String}(undef, numPhrases) 
    unigramFreqs = Vector{Int32}(undef, numPhrases)
    stopWordIDs = Set{Int32}()
    for (phrase, phraseID) in vocab
        phrases[phraseID] = phrase
        unigramFreqs[phraseID] = 0
        if phrase in stopWords
            push!(stopWordIDs, phraseID)
        end
    end
    bigramFreqs = Dict{Tuple{Int32,Int32},Int32}()
    bigramOccurrences = Dict{Tuple{Int32,Int32},Set{Int32}}()
    for (groupID, tokIDGroup) in enumerate(tokIDGroups)
        groupSize = length(tokIDGroup)
        for (i, tokID) in enumerate(tokIDGroup)
            unigramFreqs[tokID] += 1
            if i + 1 <= groupSize
                bigram = (tokID, tokIDGroup[i + 1])
                if bigram[1] in stopWordIDs || bigram[2] in stopWordIDs
                    continue
                end
                if haskey(bigramFreqs, bigram)
                    bigramFreqs[bigram] += 1
                    push!(bigramOccurrences[bigram], groupID)
                else
                    bigramFreqs[bigram] = 1
                    bigramOccurrences[bigram] = Set{Int32}([groupID])
                end
            end
        end
    end

    for iter in 1:maxIter
        if iter % 1000 == 0
            println("word2phrase iter $iter")
        end

        bestScore = 0.0
        bestBigram = (0, 0)
        for (bigram, freq) in bigramFreqs
            if freq <= minFreq
                continue
            end
            score = (freq - minFreq) / (unigramFreqs[bigram[1]] * unigramFreqs[bigram[2]])
            if score > bestScore
                bestScore = score
                bestBigram = bigram
            end
        end
        if bestScore == 0.0
            break
        end

        newPhrase = phrases[bestBigram[1]] * "-" * phrases[bestBigram[2]]
        newPhraseID = if !haskey(vocab, newPhrase)
            push!(phrases, newPhrase)
            length(phrases)
        else 
            vocab[newPhrase]
        end
        vocab[newPhrase] = newPhraseID
        
        push!(unigramFreqs, 0)    
        occurrences = bigramOccurrences[bestBigram]
        for groupID in occurrences
            # update group in tokIDGroups
            oldTokIDGroup = tokIDGroups[groupID]
            newTokIDGroup = Vector{Int32}()
            oldGroupSize = length(oldTokIDGroup)
            i = 1
            while i <= oldGroupSize
                if oldTokIDGroup[i] == bestBigram[1] && i + 1 <= oldGroupSize
                    if oldTokIDGroup[i + 1] == bestBigram[2]
                        push!(newTokIDGroup, newPhraseID)
                        i += 2
                    else
                        push!(newTokIDGroup, oldTokIDGroup[i])
                        i += 1
                    end
                else
                    push!(newTokIDGroup, oldTokIDGroup[i])
                    i += 1
                end
            end
            tokIDGroups[groupID] = newTokIDGroup

            # subtract freqs from unigramFreqs and bigramFreqs
            oldBigramSet = Set{Tuple{Int32,Int32}}()
            for (i, tokID) in enumerate(oldTokIDGroup)
                unigramFreqs[tokID] -= 1
                if i < oldGroupSize
                    bigram = (tokID, oldTokIDGroup[i + 1])
                    if bigram[1] in stopWordIDs || bigram[2] in stopWordIDs
                        continue
                    end
                    bigramFreqs[bigram] -= 1
                    push!(oldBigramSet, bigram)
                end
            end

            # add freqs to unigramFreqs and bigramFreqs
            newGroupSize = length(newTokIDGroup)
            newBigramSet = Set{Tuple{Int32,Int32}}()
            for (i, tokID) in enumerate(newTokIDGroup)
                unigramFreqs[tokID] += 1
                if i < newGroupSize
                    bigram = (tokID, newTokIDGroup[i + 1])
                    if bigram[1] in stopWordIDs || bigram[2] in stopWordIDs
                        continue
                    end
                    if haskey(bigramFreqs, bigram)
                        bigramFreqs[bigram] += 1
                    else
                        bigramFreqs[bigram] = 1
                    end
                    push!(newBigramSet, bigram)
                end
            end

            # removed dispearing bigram occurences 
            for bigram in oldBigramSet
                if bigram ∉ newBigramSet
                    delete!(bigramOccurrences[bigram], groupID)
                end
            end

            # insert new bigram occurrences
            for bigram in newBigramSet
                if bigram ∉ oldBigramSet
                    if haskey(bigramOccurrences, bigram)
                        push!(bigramOccurrences[bigram], groupID)
                    else
                        bigramOccurrences[bigram] = Set{Int32}([groupID])
                    end
                end
            end
        end
    end 
end

function detectPhrases(titles::Vector{String}, fileName::String)
    numCPUs = length(Sys.cpu_info())
    jobs = Channel{Tuple{Int,Int}}(numCPUs)
    Segments = Vector{Tuple{Int,Int}}
    TokenGroups = Vector{Vector{StemmedToken}}
    SegmentedTokenGroups = Tuple{Segments, TokenGroups}
    results = Channel{SegmentedTokenGroups}()
    function runJob(iCPU::Int)
        stemmer = Stemmer("english")
        mySegments = Segments()
        myTokGroups = TokenGroups()
        for job in jobs
            push!(mySegments, job)
            for title in titles[job[1]:job[2]]
                toksOfTitle = unhyphenAndStem(stemmer, lowercase.(tokenize(title))) 
                push!(myTokGroups, toksOfTitle) 
            end
        end
        put!(results, (mySegments, myTokGroups))
        # println("$iCPU done")
    end
    function makeJobs(batchSize::Int)
        n = length(titles)
        for i in 1:batchSize:n
            put!(jobs, (i, min(i + batchSize - 1, n)))
        end
    end
    bind(jobs, @async makeJobs(1000))
    for iCPU in 1:numCPUs
        Threads.@spawn runJob(iCPU)
    end
    tokGroups = TokenGroups(undef, length(titles))
    for iCPU in 1:numCPUs
        segmentedTokenGroups = take!(results)
        segments = segmentedTokenGroups[1]
        offset = 1
        for segment in segments
            tokGroups[segment[1]:segment[2]] = segmentedTokenGroups[2][offset:offset+segment[2]-segment[1]]
            offset += segment[2]-segment[1] + 1 
        end
    end

    vocab, strongLinkForest, tokIDGroups = getVocab(tokGroups)
    println("old vocab size: ", length(vocab))
    mergeWithStrongLinks!(tokIDGroups, vocab, strongLinkForest)
    println("new vocab size: ", length(vocab))
    word2Phrase!(tokIDGroups, vocab, 30000, 5)
    println("vocab size after word2phrase: ", length(vocab))
    saveTrainingFile(fileName, tokIDGroups, vocab)
end

nipsTitles = loadTitles("DualAspectTopicRepresentation/data/NIPS.txt")
detectPhrases(nipsTitles, "DualAspectTopicRepresentation/workspace/NIPS.awe.training.txt")
trainWordEmbedding(
    "DualAspectTopicRepresentation/workspace/NIPS.awe.training.txt", 
    "DualAspectTopicRepresentation/workspace/NIPS.awe.model.txt", 
    200, 3)
