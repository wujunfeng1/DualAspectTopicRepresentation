using TextAnalysis
function loadYearsAndTitles(fileName::String)::Vector{Tuple{Int,String}}
    yearTitles = Vector{Tuple{Int,String}}()
    file = open(fileName)
    for line in readlines(file)
        if line == ""
            continue
        end
        fields = split(line, ". ")
        @assert length(fields) == 2
        ymd = split(fields[1], "-")
        year = parse(Int, ymd[1])
        title = fields[2]
        push!(yearTitles, (year, title))
    end
    yearTitles
end

function unhyphenAndStem(stemmer::Stemmer, toks::Vector{String})::Vector{String}
    result = Vector{String}()
    for tok in toks
        if tok == ""
            continue
        end
        try
            tok = replace(tok, "/" => "-or-")
            while startswith(tok, "-")
                tok = tok[2:end]
            end
            while endswith(tok, "-")
                tok = tok[1:end - 1]
            end
            subToks = split(tok, "-")
            stemmedSubToks = stem(stemmer, subToks)
            for subTok in stemmedSubToks
                push!(result, subTok)
            end
        catch
            push!(result, tok)
        end
    end
    result
end

YearTokens = Tuple{Int,String,Vector{String}}
function tokenizeYearTitles(yearTitles::Vector{Tuple{Int,String}})::Vector{YearTokens}
    numCPUs = length(Sys.cpu_info())
    jobs = Channel{Tuple{Int,Int}}(numCPUs)
    Segments = Vector{Tuple{Int,Int}}
    SegmentedYearTokens = Tuple{Segments,Vector{Vector{YearTokens}}}
    jobOutputs = Channel{SegmentedYearTokens}(numCPUs)

    function runJob(iCPU::Int)
        stemmer = Stemmer("english")
        mySegments = Segments()
        myYearTokens = Vector{Vector{YearTokens}}()
        for job in jobs
            push!(mySegments, job)
            jobYearTokens = Vector{YearTokens}()
            for yearTitle in yearTitles[job[1]:job[2]]
                year = yearTitle[1]
                title = replace(lowercase(yearTitle[2]), "—" => "-")
                toksOfTitle = tokenize(title) 
                toksOfTitle = unhyphenAndStem(stemmer, toksOfTitle)
                push!(jobYearTokens, (year, title, toksOfTitle))
            end
            push!(myYearTokens, jobYearTokens)
        end
        put!(jobOutputs, (mySegments, myYearTokens))
    end

    function makeJobs(batchSize::Int)
        n = length(yearTitles)
        for i in 1:batchSize:n
            put!(jobs, (i, min(i + batchSize - 1, n)))
        end
    end
    bind(jobs, @async makeJobs(1000))
    for iCPU in 1:numCPUs
        Threads.@spawn runJob(iCPU)
    end

    yearTitleTokens = Vector{YearTokens}(undef, length(yearTitles))
    for iCPU in 1:numCPUs
        segmentedYearTokens = take!(jobOutputs)
        segments = segmentedYearTokens[1]
        for (segID, segment) in enumerate(segments)
            yearTitleTokens[segment[1]:segment[2]] = segmentedYearTokens[2][segID]
        end
    end
    yearTitleTokens
end

function saveYearTitleTokens(fileName::String, yearTitleTokens::Vector{YearTokens})
    file = open(fileName, "w")
    for yearTokens in yearTitleTokens
        year = yearTokens[1]
        title = yearTokens[2]
        tokens = yearTokens[3]
        print(file, "$year. $title.")
        for token in tokens
            print(file, " $token")
        end
        print(file, "\n")
    end
    close(file)
end

for i in 0:106
    println("processing title block $i ...")
    yearTitles = loadYearsAndTitles("SCC2021/MAGTitles/MAGEnTitles.$i.txt")
    yearTitleTokens = tokenizeYearTitles(yearTitles)
    saveYearTitleTokens("TokenizedMAGTitles/MAGTitles.$i.txt", yearTitleTokens)
end