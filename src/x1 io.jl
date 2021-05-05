function loadTitles(fileName::String)::Vector{String}
    titles = Vector{String}()
    file = open(fileName)
    for line in readlines(file)
        if line == ""
            continue
        end
        title = split(line, ". ")[end]
        push!(titles, title)
    end
    titles
end

function saveTrainingFile(fileName::String, tokIDGroups::Vector{Vector{Int32}}, vocab::Dict{String,Int32})
    numPhrases = length(vocab)
    phrases = Vector{String}(undef, numPhrases)
    for (phrase, phraseID) in vocab
        phrases[phraseID] = phrase 
    end
    file = open(fileName, "w")
    for tokIDGroup in tokIDGroups
        text = ""
        for tokID in tokIDGroup
            text *= " " * phrases[tokID]
        end
        if length(text) > 0
            text = text[2:end]
        end
        println(file, text)
    end
    close(file)
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

function saveNonStopWords(fileName::String, nonStopWords::Vector{String})
    file = open(fileName, "w")
    for phrase in nonStopWords
        println(file, phrase)
    end
    close(file)
end

function saveAspects(fileName::String, aspects::Vector{Set{Int}}, nonStopWords::Vector{String}, cardinalities::Vector{Int})
    file = open(fileName, "w")
    for (aspectID, aspect) in enumerate(aspects)
        for phraseID in aspect
            phrase = nonStopWords[phraseID]
            card = cardinalities[phraseID]
            println(file, "$aspectID $card $phrase")
        end
    end
    close(file)
end

function saveAspects(fileName::String, aspects1::Vector{Set{Int}}, aspects2::Vector{Set{Int}}, nonStopWords::Vector{String}, cardinalities::Vector{Int})
    file = open(fileName, "w")
    for (aspectID2, aspect2) in enumerate(aspects2)
        for aspectID1 in aspect2
            aspect1 = aspects1[aspectID1]
            for phraseID in aspect1
                phrase = nonStopWords[phraseID]
                card = cardinalities[phraseID]
                println(file, "$aspectID2 $card $phrase")
            end
        end
    end
    close(file)
end

function saveAspects(fileName::String, aspects1::Vector{Set{Int}}, aspects2::Vector{Set{Int}}, aspects3::Vector{Set{Int}}, nonStopWords::Vector{String}, cardinalities::Vector{Int})
    file = open(fileName, "w")
    for (aspectID3, aspect3) in enumerate(aspects3)
        for aspectID2 in aspect3
            aspect2 = aspects2[aspectID2]
            for aspectID1 in aspect2
                aspect1 = aspects1[aspectID1]
                for phraseID in aspect1
                    phrase = nonStopWords[phraseID]
                    card = cardinalities[phraseID]
                    println(file, "$aspectID3 $card $phrase")
                end
            end
        end
    end
    close(file)
end