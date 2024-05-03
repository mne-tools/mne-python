const toggleVisibility = (className, button) => {
  const elements = document.querySelectorAll(`.${className}`)

  elements.forEach(element => {
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

  // Take care of the button content and tooltip
  button.innerText = button.innerText === "➖" ? "➕" : "➖"
  button.title = button.title === 'Hide section' ? 'Show section' : 'Hide section'
  button.classList.toggle("collapsed")
}
