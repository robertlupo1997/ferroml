# Common AI Mistakes & Fixes

AI makes predictable mistakes. Here's how to recognize and correct them.

---

## Layout Issues

### Problem: Everything in a Single Column
**What you see:** Elements stacked vertically when they should be side-by-side
**Why it happens:** AI defaults to simple layouts
**Fix phrase:** 
```
"Put [X] and [Y] side by side"
"Use a two-column layout for [section]"
"The cards should be in a grid, 3 per row"
```

### Problem: Content Touching Edges
**What you see:** Text or elements flush against screen edges
**Why it happens:** Missing padding/margin
**Fix phrase:**
```
"Add padding to the [container/section]"
"There should be more space around the content"
"Add margins to the main container"
```

### Problem: Not Responsive on Mobile
**What you see:** Elements overlap or overflow on small screens
**Why it happens:** AI forgets to add responsive styles
**Fix phrase:**
```
"Make this responsive for mobile"
"On mobile, stack these vertically"
"The sidebar should collapse into a hamburger menu on mobile"
```

### Problem: Inconsistent Spacing
**What you see:** Some gaps bigger than others randomly
**Why it happens:** Mixed spacing values
**Fix phrase:**
```
"Use consistent spacing throughout"
"All the gaps between cards should be the same"
"Standardize the padding in all sections"
```

---

## Styling Issues

### Problem: Generic/Boring Design
**What you see:** Looks like a bootstrap template
**Why it happens:** AI plays it safe
**Fix phrase:**
```
"This looks too generic. Make it more [modern/bold/playful/professional]"
"Add more visual interest"
"Use a more distinctive color palette"
"Make it look like [reference URL]"
```

### Problem: Poor Color Contrast
**What you see:** Text hard to read against background
**Why it happens:** AI picks colors that look nice but aren't accessible
**Fix phrase:**
```
"The text is hard to read, improve the contrast"
"Use a darker text color on that background"
"The button text should be white, not gray"
```

### Problem: Too Many Colors
**What you see:** Rainbow explosion
**Why it happens:** AI gets creative without restraint
**Fix phrase:**
```
"Use a simpler color palette - max 3 colors"
"Stick to [primary color] and [accent color] only"
"Remove the extra colors, keep it minimal"
```

### Problem: Tiny Text
**What you see:** Text too small to read comfortably
**Why it happens:** AI uses default sizes
**Fix phrase:**
```
"Make the body text bigger - at least 16px"
"The headlines need more size contrast"
"Increase font size throughout"
```

---

## Functionality Issues

### Problem: Button Does Nothing
**What you see:** Clicking has no effect
**Why it happens:** AI forgot to wire up the handler
**Fix phrase:**
```
"The [button name] button doesn't do anything. It should [action]"
"Connect the button to the [function]"
"When I click [X], it should [Y]"
```

### Problem: Form Submits Empty
**What you see:** Can submit without filling fields
**Why it happens:** Missing validation
**Fix phrase:**
```
"Add validation - [field] is required"
"Don't allow empty submissions"
"Show an error if [field] is empty"
```

### Problem: No Feedback After Action
**What you see:** Click something, nothing happens visually
**Why it happens:** Missing success/loading states
**Fix phrase:**
```
"Show a success message after [action]"
"Add a loading spinner while submitting"
"Give feedback when the user [does action]"
```

### Problem: Error Not Clear
**What you see:** Generic "Error" message or red text that doesn't explain
**Why it happens:** AI uses placeholder error messages
**Fix phrase:**
```
"The error message should explain what went wrong"
"Tell the user how to fix the error"
"Be specific: 'Email already exists' not just 'Error'"
```

---

## Component Issues

### Problem: Missing Empty State
**What you see:** Blank space when there's no data
**Why it happens:** AI only handles happy path
**Fix phrase:**
```
"What happens when there's no [data]? Add an empty state"
"Show a message when the list is empty"
"Add a placeholder for when there are no [items]"
```

### Problem: Missing Loading State
**What you see:** Content jumps in suddenly
**Why it happens:** AI doesn't simulate async behavior
**Fix phrase:**
```
"Add a loading skeleton while data loads"
"Show a spinner before the content appears"
"Add loading states to the [component]"
```

### Problem: Broken Images
**What you see:** Broken image icons or missing images
**Why it happens:** AI uses placeholder URLs that don't exist
**Fix phrase:**
```
"Use actual placeholder images from picsum.photos or similar"
"Add fallback images when loading fails"
"Replace the broken images with working ones"
```

### Problem: Overflow/Cut-off Content
**What you see:** Text or content getting cut off
**Why it happens:** Fixed heights without overflow handling
**Fix phrase:**
```
"The text is getting cut off, allow it to wrap"
"Add scrolling to the [container] if content overflows"
"Don't truncate the [content], let it expand"
```

---

## Code Quality Issues

### Problem: Repeated Code
**What you see:** Same thing copied multiple times
**Why it happens:** AI doesn't refactor
**Fix phrase:**
```
"Extract this into a reusable component"
"Don't repeat this - create a shared [component/function]"
"This pattern is repeated, make it DRY"
```

### Problem: Hardcoded Values
**What you see:** Specific numbers, texts, or URLs in code
**Why it happens:** AI uses example data directly
**Fix phrase:**
```
"Make [value] configurable, don't hardcode it"
"Move these values to a config file"
"Use variables instead of hardcoded values"
```

### Problem: No Types (TypeScript)
**What you see:** `any` types everywhere
**Why it happens:** AI takes shortcuts
**Fix phrase:**
```
"Add proper TypeScript types"
"Define an interface for [data structure]"
"Remove the 'any' types and use proper types"
```

---

## Quick Fix Reference

| Problem | One-Line Fix |
|---------|-------------|
| Too cramped | "Add more whitespace and padding" |
| Too spread out | "Reduce the spacing, make it more compact" |
| Looks boring | "Make it more visually interesting" |
| Too busy | "Simplify - remove decorative elements" |
| Not working | "The [feature] is broken, please fix it" |
| Wrong behavior | "When I [action], it should [expected] but it [actual]" |
| Missing something | "Add [missing element]" |
| Ugly on mobile | "Fix the mobile layout" |

---

## Prevention Tips

Tell AI upfront to avoid common issues:

```
When building this:
- Use consistent spacing (8px grid system)
- Ensure all text has good contrast (WCAG AA minimum)
- Make everything responsive from the start
- Add loading and empty states
- Include form validation
- Use TypeScript with proper types
- Test all buttons work
```

---

## When AI Gets Stuck in a Loop

Sometimes AI keeps making the same mistake. Try:

1. **Be more specific**
   - Bad: "Fix the layout"
   - Good: "The sidebar should be exactly 250px wide and fixed position"

2. **Reference working code**
   - "Copy how [other component] handles this"

3. **Simplify**
   - "Remove the [feature] for now, let's get the basics working first"

4. **Start fresh**
   - "Delete this component and start over with a simpler approach"

5. **Show example**
   - "Here's what I mean: [paste code or URL]"
