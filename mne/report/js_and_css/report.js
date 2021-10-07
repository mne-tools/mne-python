/* We need to refresh the scroll spy after (un)hiding elements */
const refreshScrollSpy = () =>{
  const dataSpyList = [].slice.call(document.querySelectorAll('[data-bs-spy="scroll"]'));
  dataSpyList.forEach((dataSpyEl) => {
    bootstrap.ScrollSpy.getInstance(dataSpyEl)
    .refresh()
  })
  console.log('scrollspy refreshed!');  // TODO remove debugging output
}

/* Show or hide elements based on their tag */
const toggleTagVisibility = (tagName) => {
  const tag = tags.find((element) => {
    return element.name === tagName;
  });
  tag.visible = !tag.visible;

  const hiddenTagNames = tags.map((tag) => {
    if (tag.visible) {
      return
    } else {
      return tag.name
    }
  });
  const elements = $(`[data-mne-tags~="${tagName}"]`);
  elements.each((i) => {
    const currentElement = elements[i];
    const tagValuesOfCurrentElement = currentElement.getAttribute('data-mne-tags');

    // TODO This can probably be refactored to not use a Set.
    const tagNamesOfCurrentElement = new Set(tagValuesOfCurrentElement.match(/\S+/g));  // non-whitespace
    const visibleTagNamesOfCurrentElement = new Set(
      [...tagNamesOfCurrentElement].filter(e => !hiddenTagNames.includes(e))
    );

    if (visibleTagNamesOfCurrentElement.size === 0) {  // hide
      $(currentElement).slideToggle('fast', () => {
        $(currentElement).addClass('d-none');
      });
    } else if ($(currentElement).hasClass('d-none')) {  // show
      $(currentElement).removeClass('d-none');
      $(currentElement).slideToggle('fast');
    }
  })

  const tagBadgeElements = document.querySelectorAll(`span.badge[data-mne-tag~="${tagName}"]`);
  tagBadgeElements.forEach((badgeElement) => {
    if (tag.visible) {
      badgeElement.removeAttribute('data-mne-tag-hidden');
      $(badgeElement).removeClass('bg-secondary');
      $(badgeElement).addClass('bg-primary');
    } else {
      badgeElement.setAttribute('data-mne-tag-hidden', true);
      $(badgeElement).removeClass('bg-primary');
      $(badgeElement).addClass('bg-secondary');
    }
  })

  refreshScrollSpy();
}


/* Gather all available tags and expose them in the global namespace */
let tags = [];  // array of objects

const  gatherTags = () => {
  // only consider top-level elements
  const taggedElements = document.querySelectorAll("#content > div[data-mne-tags]");

  taggedElements.forEach((element) => {
      const value = element.getAttribute('data-mne-tags');
      const tagNames = value.match(/\S+/g);  // non-whitespace
      tagNames.forEach((tagName) => {
        const existingTag = tags.find((element) => {
            return element.name === tagName;
        })

        if (existingTag === undefined) {
          const tag = {
            name : tagName,
            visible: true,
            count: 1
          };
          tags.push(tag);
        } else {
          existingTag.count = existingTag.count + 1;
        }
      })
  })
}

/* Badges do display the tag count */
const updateTagCountBadges = () => {
  const menuEntries = document
    .querySelectorAll("#filter-by-tags-dropdown-menu > ul > li > label[data-mne-tag]")

    menuEntries.forEach((menuEntry) => {
      const tagName = menuEntry.getAttribute('data-mne-tag');
      const tag = tags.find((tag) => {
        return tag.name === tagName;
      })
      const tagCount = tag.count;

      const tagCountBadge = menuEntry.querySelector('span.badge');
      tagCountBadge.innerHTML = tagCount.toString();
    });
  }

const addFilterByTagsCheckboxEventHandlers = () => {
  // "Filter by tag" checkbox event handling
  const selectAllTagsCheckboxLabel = document
    .querySelector('#selectAllTagsCheckboxLabel');
  const filterByTagsDropdownMenuLabels = document
    .querySelectorAll("#filter-by-tags-dropdown-menu > ul > li > label[data-mne-tag]")

  filterByTagsDropdownMenuLabels.forEach((label) => {
    // Prevent dropdown menu from closing when clicking on a tag checkbox label
    label.addEventListener("click", (e) => { 
      e.stopPropagation();
    })

    // Show / hide content if a tag checkbox value has changed
    const tagName = label.getAttribute("data-mne-tag");
    const checkbox = label.querySelector("input");
    checkbox.addEventListener("change", () => {
      toggleTagVisibility(tagName);
    })
  })

  // "Select all" checkbox
  selectAllTagsCheckboxLabel.addEventListener("click", (e) => {
    e.stopPropagation();
  })
  const selectAllTagsCheckbox = selectAllTagsCheckboxLabel.querySelector('input');

  selectAllTagsCheckbox.addEventListener("change", (e) => {
    const selectAllCheckboxStatus = e.target.checked;

    filterByTagsDropdownMenuLabels.forEach((element) => {
      const checkbox = element.querySelector('input');
      if (checkbox.checked !== selectAllCheckboxStatus) {
        checkbox.checked = selectAllCheckboxStatus

        // we need to manually trigger the change event
        const changeEvent = new Event('change');
        checkbox.dispatchEvent(changeEvent);
      }
    })
  });
}

const _handleTocLinkClick = (e) => {
    e.preventDefault();

    const topBarHeight = document.querySelector('#top-bar').scrollHeight
    const margin = 30 + topBarHeight;
  
    const tocLinkElement = e.target;
    const targetDomId = tocLinkElement.getAttribute('href');
    const targetElement = document.querySelector(targetDomId);
    const top = $(targetElement).offset().top;
    window.scrollTo(0, top - margin);
}

const fixScrollingForTocLinks = () => {
  const tocLinkElements = document.querySelectorAll('#toc-navbar > a');

  tocLinkElements.forEach((element) => {
    element.removeEventListener('click', _handleTocLinkClick)
    element.addEventListener('click', _handleTocLinkClick)
  })
}

const addSliderEventHandlers = () => {
  const accordionElementsWithSlider = document.querySelectorAll('div.accordion-item.slider');
  accordionElementsWithSlider.forEach((el) => {
    const accordionElement = el.querySelector('div.accordion-body');

    const slider = accordionElement.querySelector('input');
    // const sliderLabel = accordionElement.querySelector('label');
    const carousel = accordionElement.querySelector('div.carousel');
    slider.addEventListener('input', (e) => {
      const sliderValue = parseInt(e.target.value);
      $(carousel).carousel(sliderValue);
    })
  })
}

const fixTopMargin = () => {
  const topBarHeight = document.querySelector('#top-bar').scrollHeight
  const margin = 30 + topBarHeight;

  document.getElementById('content').style.marginTop = `${margin}px`;
  document.getElementById('toc').style.marginTop = `${margin}px`;
}


$(document).ready(() => {
  gatherTags();
  updateTagCountBadges();
  addFilterByTagsCheckboxEventHandlers();
  addSliderEventHandlers();
  fixTopMargin();
  fixScrollingForTocLinks();
  hljs.highlightAll();   // enable highlight.js
});

window.onresize = () => {
  fixTopMargin();
  refreshScrollSpy();
};

/* Show / hide all tags on keypress */
window.onkeydown = (e) => {
  if (e.code === "KeyT") {
    const selectAllTagsCheckbox = document
      .querySelector('#selectAllTagsCheckboxLabel > input');
    selectAllTagsCheckbox.checked = !selectAllTagsCheckbox.checked;

    // we need to manually trigger the change event
    const changeEvent = new Event('change');
    selectAllTagsCheckbox.dispatchEvent(changeEvent);
  }
}
