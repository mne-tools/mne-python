const toggleVisibility = (className) => {

  const elements = document.querySelectorAll(`.${className}`)

  elements.forEach(element => {
    if (element.classList.contains('repr-section-header')) {
      // Don't collapse the section header row.
       return
    }
    if (element.classList.contains('repr-element-collapsed')) {
      // Force a reflow to ensure the display change takes effect before removing the class
      element.classList.remove('repr-element-collapsed')
      element.offsetHeight // This forces the browser to recalculate layout
      element.classList.remove('repr-element-faded')
    } else {
      // Start transition to hide the element
      element.classList.add('repr-element-faded')
      element.addEventListener('transitionend', handler = (e) => {
        if (e.propertyName === 'opacity' && getComputedStyle(element).opacity === '0.2') {
          element.classList.add('repr-element-collapsed')
          element.removeEventListener('transitionend', handler)
        }
      });
    }
  });

  // Take care of button (adjust caret)
  const button = document.querySelectorAll(`.repr-section-header.${className} > th.repr-section-toggle-col > button`)[0]
  button.classList.toggle('collapsed')

  // Take care of the tooltip of the section header row
  const sectionHeaderRow = document.querySelectorAll(`tr.repr-section-header.${className}`)[0]
  sectionHeaderRow.classList.toggle('collapsed')
  sectionHeaderRow.title = sectionHeaderRow.title === 'Hide section' ? 'Show section' : 'Hide section'
}
