using WebDriver

capabilities = Capabilities("firefox")
wd = RemoteWebDriver(capabilities)
session = Session(wd)

function waitForPageLoaded(session)
    sleep(1)
    pageState = try
        script!(session, "return document.readyState")
    catch
        ""
    end
    for i = 1:30
        if pageState != "complete"
            sleep(1)
        end
    end
end

url = "https://proceedings.neurips.cc/"
navigate!(session, url)
waitForPageLoaded(session)
tableElem = Element(session, "css selector", "div.col-sm")
linkElems = Elements(tableElem, "css selector", "a")
links = [element_attr(linkElem, "href") for linkElem in linkElems]

file = open("DATR/data/NIPS.txt", "w")
for url in links
    textOfYear = split(url,"/")[end]
    navigate!(session, url)
    waitForPageLoaded(session)
    tableElem = Element(session, "css selector", "div.col")
    sublinkElems = Elements(tableElem, "css selector", "a")
    for sublinkElem in sublinkElems
        title = replace(replace(element_text(sublinkElem), "\n"=>" "), "￼"=>"")
        println(file, "$textOfYear. $title")
    end
end
close(file)