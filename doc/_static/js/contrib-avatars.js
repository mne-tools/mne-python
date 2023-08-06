async function getContribs(url) {
    result = await fetch(url);
    data = await result.json();
    return data;
}

function addCards(data, container) {
    data.forEach((entry) => {
      if (!entry.login.endsWith("[bot]")) {
        let card = document.createElement("div");
        let anchor = document.createElement("a");
        let image = document.createElement("img");
        card.setAttribute("class", "card my-1 mx-2");
        anchor.setAttribute("href", entry.html_url);
        image.setAttribute("class", "card-img contributor-avatar");
        image.setAttribute("src", entry.avatar_url);
        image.setAttribute("title", entry.login);
        image.setAttribute("alt", `Contributor avatar for ${entry.login}`);
        anchor.append(image);
        card.append(anchor);
        container.append(card);
      }
    });
}

async function putAvatarsInPage() {
    // container
    let outer = document.createElement("div");
    let title = document.createElement("p");
    let inner = document.createElement("div");
    outer.setAttribute("id", "contributor-avatars");
    outer.setAttribute("class", "container my-4");
    title.setAttribute("class", "h4 text-center font-weight-light");
    title.innerText = "Contributors";
    inner.setAttribute("class", "d-flex flex-wrap flex-row justify-content-center align-items-center");
    // GitHub API returns batches of 100 so we have to loop
    var page = 1;
    while (true) {
      data = await getContribs(
        `https://api.github.com/repos/mne-tools/mne-python/contributors?per_page=100&page=${page}`
      );
      if (!data.length) {
        break;
      }
      addCards(data, container=inner);
      page++;
    }
    // finish
    outer.append(title, inner);
    document.getElementById("institution-logos").after(outer);
}

putAvatarsInPage();
