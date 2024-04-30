const toggleVisibility = (className, button) => {
  const elements = document.querySelectorAll(`.${className}`)

  elements.forEach(element => {
    if (element.classList.contains('repr-element-hidden')) {
      // Remove display:none to start showing the element
      element.style.display = ''

      // Force a reflow to ensure the display change takes effect before removing the class
      element.offsetHeight // This forces the browser to recalculate layout

      element.classList.remove('repr-element-hidden')
    } else {
      // Start transition to hide the element
      element.classList.add('repr-element-hidden')
      element.addEventListener('transitionend', function handler(e) {
        if (e.propertyName === 'opacity' && getComputedStyle(element).opacity === '0.2') {
          element.style.display = 'none'
          element.removeEventListener('transitionend', handler)
        }
      });
    }
  });

  // Take care of the button content and tooltip
  button.innerHTML = button.innerHTML.replace(/➖|➕/g, (match) =>
    match === "➖" ? "➕" : "➖"
  )
  button.title = button.title === 'Hide section' ? 'Show section' : 'Hide section'
}
