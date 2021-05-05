using TextAnalysis

struct StemmedToken
    text::String
    stemmed::Vector{String} 
end

function unhyphenAndStem(stemmer::Stemmer, toks::Vector{String})::Vector{StemmedToken}
    result = Vector{StemmedToken}()
    for tok in toks
        if tok == ""
            continue
        end
        tok = replace(tok, "/" => "-or-")
        while startswith(tok, "-")
            tok = tok[2:end]
        end
        while endswith(tok, "-")
            tok = tok[1:end - 1]
        end
        subToks = split(tok, "-")
        stemmedToken = StemmedToken(tok, stem(stemmer, subToks))
        push!(result, stemmedToken)
    end
    result
end

IDPair = Tuple{Int32,Int32}
IDPostfixes = Set{Vector{Int32}}

function getVocab(tokGroups::Vector{Vector{StemmedToken}})::Tuple{Dict{String,Int32},Dict{IDPair,IDPostfixes},Vector{Vector{Int32}}}
    vocab = Dict{String,Int32}()
    strongLinkForest = Dict{IDPair,IDPostfixes}()
    numGroups = length(tokGroups)
    tokIDGroups = Vector{Vector{Int32}}(undef, numGroups)
    for (groupID, tokGroup) in enumerate(tokGroups)
        tokIDGroup = Vector{Int32}()
        for stemmedToken in tokGroup
            for subtok in stemmedToken.stemmed
                if haskey(vocab, subtok)
                    push!(tokIDGroup, vocab[subtok])
                    continue
                end
                subtokID = length(vocab) + 1
                vocab[subtok] = subtokID
                push!(tokIDGroup, subtokID)
            end
            numSubToks = length(stemmedToken.stemmed)
            if numSubToks > 1
                idPair = (tokIDGroup[end - numSubToks + 1], tokIDGroup[end - numSubToks + 2])
                postfix = tokIDGroup[end - numSubToks + 3:end]
                if !haskey(strongLinkForest, idPair)
                    strongLinkForest[idPair] = IDPostfixes()
                end
                push!(strongLinkForest[idPair], postfix)
            end
        end
        tokIDGroups[groupID] = tokIDGroup 
    end
    vocab, strongLinkForest, tokIDGroups
end

const stopWords = Set{String}([
    "a",
	"an",
	"and",
	"as",
	"based",
	"by",
	"for",
	"from",
	"in",
	"on",
	"of",
    "over",
	"that",
	"the",
	"this",
	"to",
	"via",
	"with",
	"without",
	"is",
	"are",
	"be",
	"you",
	"we",
	"they",
	"it",
	"your",
	"our",
	"their",
	"its",
	"what",
	"when",
	"where",
	"how",
	"do",
	"use",
	"here",
	"there",
	"using",
    "than",
	"、",
	",",
	"，",
	":",
	"：",
	".",
	"。",
	"‧",
	"!",
	"！",
	"?",
	"？",
	";",
	"；",
	"(",
	"（",
	")",
	"）",
	"'",
	"‘",
	"’",
	"\"",
	"「",
	"」",
	"“",
	"”",
	"`",
	"…",
    "</s>",
])

function getNonStopWords(wordVectors::WordVectors)::Tuple{Vector{String},Dict{String,Int}}
    nonStopWords = Vector{String}()
    nonStopWordIDs = Dict{String,Int}()
    for (wordID, word) in enumerate(wordVectors.vocab)
        if !(word in stopWords)
            push!(nonStopWords, word)
            nonStopWordIDs[word] = wordID
        end
    end
    (nonStopWords, nonStopWordIDs)
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

function getVectors(wordVectors::WordVectors, nonStopWords::Vector{String})::Array{Float64,2}
    numDims, numWords = size(wordVectors)
    numVectors = length(nonStopWords)
    result = Array{Float64,2}(undef, (numDims, numVectors))
    for i in 1:numVectors
        vector = get_vector(wordVectors, nonStopWords[i])
        result[:,i] = vector
    end
    result
end

function getPhrasePairFreqs(nonStopWords::Vector{String}, nonStopWordIDs::Dict{String,Int}, phraseIDGroups::Vector{Vector{Int}})::Dict{Tuple{Int,Int},Int}
    numPhrases = length(nonStopWords)
    idMap = Dict{Int,Int}()
    for (newID, phrase) in enumerate(nonStopWords)
        idMap[nonStopWordIDs[phrase]] = newID
    end

    pairFreqs = Dict{Tuple{Int,Int},Int}()
    for idGroup in phraseIDGroups
        numIDsInGroup = length(idGroup)
        convertedIDs = [idMap[id] for id in idGroup]
        for (j1, id1) in enumerate(convertedIDs)
            for j2 in j1 + 1:numIDsInGroup
                id2 = convertedIDs[j2]
                pair = if id1 < id2
                    (id1, id2)
                else
                    (id2, id1)
                end
                pairFreqs[pair] = get(pairFreqs, pair, 0) + 1 
            end
        end
    end

    pairFreqs
end

function mergeAspects(aspects::Vector{Set{Int}}, connMat::Array{Float64,2})::Array{Float64,2}
    numAspects = length(aspects)
    newConnMat = Array{Float64,2}(undef, (numAspects, numAspects))
    for i in 1:numAspects
        for j in 1:numAspects
            if i != j
                conn = 0.0
                for ui in aspects[i]
                    for uj in aspects[j]
                        conn += connMat[ui,uj]
                    end
                end
                newConnMat[i,j] = conn
            else
                newConnMat[i,j] = 0.0
            end
        end
    end
    newConnMat
end
