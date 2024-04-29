const toggleVisibility = (className, button) => {
  const elements = document.querySelectorAll(`.${className}`)
  elements.forEach(e => e.classList.toggle('repr-element-hidden'))

  button.innerHTML = button.innerHTML.replace(/➖|➕/g, (match) =>
    match === "➖" ? "➕" : "➖"
  )
}
