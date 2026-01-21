# Minimum Viable Knowledge for AI-Assisted Frontend Development

You don't need to be a developer to direct AI effectively. This guide teaches you the **minimum** needed to review AI work and give clear corrections.

---

## Level 1: Survival Skills (30 minutes)

### Reading Error Messages

When something breaks, look for these patterns:

| Error Pattern | What It Means | Tell AI |
|---------------|---------------|---------|
| `Cannot find module 'X'` | Missing package | "Install the missing package X" |
| `TypeError` | Code logic error | "Fix the type error on line N" |
| `SyntaxError` | Typo in code | "There's a syntax error, please fix it" |
| `EADDRINUSE` | Port already used | "Use a different port" |
| `Network Error` | Connection problem | "Check if the API URL is correct" |
| `404 Not Found` | Missing page/file | "The route/file doesn't exist, create it" |

### Using Browser DevTools

Press **F12** to open DevTools. Three tabs matter:

1. **Console** - Shows errors (red = bad, yellow = warning)
2. **Elements** - Shows HTML structure (right-click anything → "Inspect")
3. **Network** - Shows what's loading (red = failed request)

### Basic File Types

| Extension | What It Is |
|-----------|-----------|
| `.tsx` / `.jsx` | React components (the building blocks) |
| `.ts` / `.js` | Regular code logic |
| `.css` | Styling (colors, sizes, spacing) |
| `.html` | Page structure |
| `.json` | Data and configuration |

---

## Level 2: Direction Skills (2 hours)

### HTML Elements (The Building Blocks)

```
<div>      = Container/box (holds other things)
<button>   = Clickable button
<input>    = Text field for typing
<a>        = Link to another page
<img>      = Image
<h1>...<h6> = Headings (h1 is biggest)
<p>        = Paragraph of text
<ul>/<li>  = Bullet list
<form>     = Group of inputs
<nav>      = Navigation menu
<header>   = Top section
<footer>   = Bottom section
```

### CSS Properties (The Styling)

```
color: blue          → Text color
background: red      → Background color
font-size: 20px      → Text size
padding: 10px        → Space INSIDE element
margin: 10px         → Space OUTSIDE element
border: 1px solid    → Border around element
width: 100px         → How wide
height: 100px        → How tall
display: flex        → Layout system
justify-content      → Horizontal alignment
align-items          → Vertical alignment
```

### React Concepts (The Framework)

```
Component  = Reusable piece (like a LEGO block)
Props      = Inputs passed to component
State      = Data that can change
onClick    = When user clicks
onChange   = When user types
useEffect  = When something should happen automatically
```

---

## Level 3: Correction Phrases

Copy-paste these when giving feedback to AI:

### Layout Problems

```
"Move the [X] to the left/right/top/bottom"
"Put [X] next to [Y]"
"Center [X] on the page"
"Add more space between [X] and [Y]"
"Make [X] take up the full width"
"Stack these vertically instead of horizontally"
"Create a two-column layout"
"Add a sidebar on the left"
```

### Style Problems

```
"Make [X] bigger/smaller"
"Change the color of [X] to [color]"
"Make the background darker/lighter"
"Use a different font"
"Add rounded corners to [X]"
"Remove the border from [X]"
"Add a shadow to [X]"
"Make it look more like [URL]"
```

### Functionality Problems

```
"When I click [X], it should [action]"
"The button doesn't do anything"
"The form doesn't submit"
"Show an error message if [condition]"
"Save this data when I click [X]"
"Load the data when the page opens"
"Redirect to [page] after [action]"
"Add a loading spinner while waiting"
```

### Content Problems

```
"Change the text to say [X]"
"Add a heading that says [X]"
"Replace the placeholder image with [description]"
"Add a link to [URL]"
"Remove the [element]"
"Duplicate this [element]"
```

---

## How to Review AI Screenshots

When AI shows you a screenshot, check:

1. **Layout** - Is everything in the right place?
2. **Readability** - Can you read all the text?
3. **Clickable things** - Do buttons/links look clickable?
4. **Empty states** - What shows when there's no data?
5. **Consistency** - Does everything match visually?

If something's wrong, describe it naturally:
- ✅ "The button is too small and hard to see"
- ✅ "The colors are too bright, make it more professional"
- ✅ "I can't tell what's clickable"
- ❌ "Fix the CSS" (too vague)

---

## Quick Reference Card

### When Something Looks Wrong
```
Layout:    "Move [X] to [position]" or "Put [X] inside [Y]"
Size:      "Make [X] bigger/smaller"
Color:     "Change [X] to [color]"
Spacing:   "Add more space around [X]"
Missing:   "Add a [element] that shows [content]"
Extra:     "Remove the [element]"
```

### When Something Doesn't Work
```
No action:     "When I click [X], nothing happens. Make it [action]"
Wrong action:  "Clicking [X] does [wrong thing]. It should [right thing]"
Error:         "I see an error: [paste the error]"
Crash:         "The page went blank after [action]"
```

### When You Want to Copy Another Site
```
"Make it look like [URL]"
"Copy the layout from [URL] but use our colors"
"The navigation should work like [URL]"
"Use a similar card design to [URL]"
```

---

## Learning Path

As you work with AI, you'll naturally learn:

| Week | You'll Understand |
|------|------------------|
| 1 | How to describe what you see vs. what you want |
| 2 | Basic HTML element names |
| 3 | How colors and spacing work |
| 4 | How components fit together |
| 8 | How to read simple code |
| 12 | How to make small fixes yourself |

**You don't need to learn everything upfront.** Just start building and learn as you go.
