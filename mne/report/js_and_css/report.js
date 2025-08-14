/* We need to refresh the scroll spy after (un)hiding elements */
const refreshScrollSpy = () =>{
  const dataSpyList = [].slice.call(document.querySelectorAll('[data-bs-spy="scroll"]'));
  dataSpyList.forEach((dataSpyEl) => {
    bootstrap.ScrollSpy.getInstance(dataSpyEl)
    .refresh()
  })
}

const propagateScrollSpyURL = () => {
  window.addEventListener('activate.bs.scrollspy', (e) => {
    history.replaceState({}, "", e.relatedTarget);
  });
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
        currentElement.classList.add('d-none');
      });
    } else if ($(currentElement).hasClass('d-none')) {  // show
      currentElement.classList.remove('d-none');
      $(currentElement).slideToggle('fast');
    }
  })

  const tagBadgeElements = document.querySelectorAll(`span.badge[data-mne-tag~="${tagName}"]`);
  tagBadgeElements.forEach((badgeElement) => {
    if (tag.visible) {
      badgeElement.removeAttribute('data-mne-tag-hidden');
      badgeElement.classList.remove('bg-secondary');
      badgeElement.classList.add('bg-primary');
    } else {
      badgeElement.setAttribute('data-mne-tag-hidden', true);
      badgeElement.classList.remove('bg-primary');
      badgeElement.classList.add('bg-secondary');
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

/* Avoid top of content getting hidden behind navbar after clicking on a TOC
   link */
const _handleTocLinkClick = (e) => {
    e.preventDefault();

    const topBarHeight = document.querySelector('#top-bar').scrollHeight
    const margin = 30 + topBarHeight;

    const tocLinkElement = e.target;
    const targetDomId = tocLinkElement.getAttribute('href');
    const targetElement = document.querySelector(targetDomId);
    const top = targetElement.getBoundingClientRect().top + window.scrollY;
 
    // Update URL to reflect the current scroll position.
    // We use history.pushState to change the URL without causing the browser to scroll.
    history.pushState(null, "", targetDomId);

    // Now scroll to the correct position.
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

    // Allow focussing the slider with a click on the slider or carousel, so keyboard
    // controls (left / right arrow) can be enabled.
    // This also appears to be the only way to focus the slider in Safari:
    // https://itnext.io/fixing-focus-for-safari-b5916fef1064?gi=c1b8b043fa9b
    slider.addEventListener('click', () => {
      slider.focus({preventScroll: true})
    })
    carousel.addEventListener('click', () => {
      slider.focus({preventScroll: true})
    })
  })
}

/* Avoid top of content gets hidden behind the top navbar */
const fixTopMargin = () => {
  const topBarHeight = document.querySelector('#top-bar').scrollHeight
  const margin = 30 + topBarHeight;

  document.getElementById('content').style.marginTop = `${margin}px`;
  document.getElementById('toc').style.marginTop = `${margin}px`;
}

/* Show / hide all tags on keypress */
const _globalKeyHandler = (e) => {
  if (e.code === "KeyT") {
    const selectAllTagsCheckbox = document
      .querySelector('#selectAllTagsCheckboxLabel > input');
    selectAllTagsCheckbox.checked = !selectAllTagsCheckbox.checked;

    // we need to manually trigger the change event
    const changeEvent = new Event('change');
    selectAllTagsCheckbox.dispatchEvent(changeEvent);
  }
}

const enableGlobalKeyHandler = () => {
  window.onkeydown = (e) => _globalKeyHandler(e);
}

const disableGlobalKeyHandler = () => {
  window.onkeydown = null;
}

/* Disable processing global key events when a search box is active */
const disableGlobalKeysInSearchBox = () => {
  const searchBoxElements = document.querySelectorAll('input.search-input');
  searchBoxElements.forEach((el) => {
    el.addEventListener('focus', () => disableGlobalKeyHandler());
    el.addEventListener('blur', () => enableGlobalKeyHandler());
  })
}

/* Run once all content is fully loaded. */
window.addEventListener('load', () => {
  gatherTags();
  updateTagCountBadges();
  addFilterByTagsCheckboxEventHandlers();
  addSliderEventHandlers();
  fixTopMargin();
  fixScrollingForTocLinks();
  hljs.highlightAll();   // enable highlight.js
  disableGlobalKeysInSearchBox();
  enableGlobalKeyHandler();
  propagateScrollSpyURL();
});

/* Resizing the window throws off the scroll spy and top-margin handling. */
window.onresize = () => {
  fixTopMargin();
  refreshScrollSpy();
};
