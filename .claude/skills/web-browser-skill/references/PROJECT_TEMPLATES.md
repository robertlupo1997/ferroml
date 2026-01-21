# Project Templates

Progressive templates from simple to complex. Each builds on skills from the previous.

---

## Template 1: Landing Page (Beginner)

**Complexity:** ⭐ (1/5)  
**Time:** 1-2 hours  
**Skills learned:** Layout, styling, responsive design

### What You'll Build
A single-page marketing site with:
- Hero section with headline and CTA button
- Features section with 3-4 cards
- Testimonials or social proof
- Footer with links

### Prompt to Start
```
Create a landing page for [YOUR PRODUCT/SERVICE].

Include:
- A hero section with a bold headline: "[YOUR HEADLINE]"
- A call-to-action button that says "[BUTTON TEXT]"
- A features section showing 3 benefits: [BENEFIT 1], [BENEFIT 2], [BENEFIT 3]
- A testimonial section
- A simple footer

Style: [modern/minimal/playful/professional]
Colors: [describe colors or say "suggest something"]
```

### Correction Phrases for This Project
```
"Make the hero section taller"
"The headline needs more impact"
"Add more contrast between sections"
"The CTA button doesn't stand out enough"
"Make it responsive for mobile"
```

### Success Criteria
- [ ] Looks good on desktop (1440px width)
- [ ] Looks good on mobile (375px width)
- [ ] All text is readable
- [ ] Button clearly looks clickable
- [ ] Page loads quickly

---

## Template 2: Dashboard (Intermediate)

**Complexity:** ⭐⭐⭐ (3/5)  
**Time:** 3-5 hours  
**Skills learned:** Components, data display, charts, layout systems

### What You'll Build
A data dashboard with:
- Sidebar navigation
- Header with user info
- Stat cards (numbers at a glance)
- Charts (line, bar, or pie)
- Data table
- Activity feed

### Prompt to Start
```
Create a dashboard for tracking [YOUR DATA TYPE].

Layout:
- Sidebar on the left with navigation
- Header with title and user avatar
- Main area with stats and charts

Data to display:
- Stat card 1: [METRIC NAME] showing [EXAMPLE VALUE]
- Stat card 2: [METRIC NAME] showing [EXAMPLE VALUE]
- Stat card 3: [METRIC NAME] showing [EXAMPLE VALUE]
- Line chart showing [DATA] over time
- Table showing recent [ITEMS]

Use mock/sample data for now.
Style: Clean and professional, dark/light theme
```

### Correction Phrases for This Project
```
"The sidebar should collapse on mobile"
"Add icons to the navigation items"
"The stat cards need trend indicators (up/down arrows)"
"The chart is hard to read, simplify it"
"Add hover states to the table rows"
"The data should be sortable by column"
```

### Success Criteria
- [ ] Navigation clearly shows current page
- [ ] Stats are easy to scan at a glance
- [ ] Charts are readable and meaningful
- [ ] Table handles empty states gracefully
- [ ] Responsive sidebar (collapses on mobile)

---

## Template 3: Web App with Auth (Advanced)

**Complexity:** ⭐⭐⭐⭐⭐ (5/5)  
**Time:** 8-15 hours  
**Skills learned:** Authentication, forms, state management, routing, API integration

### What You'll Build
A full web application with:
- Login / Register pages
- Protected routes (must be logged in)
- User profile page
- Settings page
- Main feature (CRUD operations)
- Logout functionality

### Prompt to Start
```
Create a web app for [YOUR APP PURPOSE].

Pages needed:
1. Login page (email + password)
2. Register page (name, email, password, confirm password)
3. Dashboard (protected - requires login)
4. [MAIN FEATURE] page for creating/viewing/editing [ITEMS]
5. Settings page for user preferences
6. Profile page

Features:
- Form validation with error messages
- Remember login state (localStorage)
- Loading states when submitting
- Success/error notifications

For now, use mock authentication (no real backend).
Store data in localStorage.
```

### Correction Phrases for This Project
```
"Show validation errors as I type, not just on submit"
"The login should redirect to dashboard on success"
"Add a 'forgot password' link"
"The protected pages should redirect to login if not authenticated"
"Add a confirmation dialog before deleting"
"Show a loading spinner while the form submits"
"The error messages are too harsh, make them friendlier"
```

### Multi-Step Approach

This project is complex. Break it into phases:

**Phase 1: Auth UI**
```
"First, let's just build the login and register pages with form validation.
No functionality yet, just the UI and validation messages."
```

**Phase 2: Auth Logic**
```
"Now add the authentication logic using localStorage.
When user logs in, store a token.
Redirect to dashboard after login.
If no token, redirect to login from protected pages."
```

**Phase 3: Main Feature**
```
"Now build the [MAIN FEATURE] page.
Users should be able to:
- Create new [ITEMS]
- View list of [ITEMS]
- Edit existing [ITEMS]
- Delete [ITEMS]
Store everything in localStorage for now."
```

**Phase 4: Polish**
```
"Let's polish the app:
- Add loading states
- Add success/error notifications
- Improve form UX
- Add empty states
- Make it responsive"
```

### Success Criteria
- [ ] Can register a new account
- [ ] Can log in with registered account
- [ ] Can't access protected pages when logged out
- [ ] Can create, read, update, delete items
- [ ] Form validation prevents bad data
- [ ] Error states are clear
- [ ] Success feedback is shown
- [ ] Data persists after page refresh

---

## Choosing Your Starting Point

| If You Want... | Start With |
|----------------|-----------|
| Quick win, learn basics | Landing Page |
| Data visualization | Dashboard |
| Full app experience | Web App with Auth |
| Portfolio piece | Dashboard or Web App |

---

## Tips for All Projects

### Before Starting
1. Find 2-3 inspiration sites
2. Screenshot what you like about them
3. List the features you need
4. Decide on a color scheme

### During Development
1. Build one section at a time
2. Check both desktop and mobile views
3. Test all interactive elements
4. Take screenshots of progress

### When Stuck
```
"Show me a screenshot of the current state"
"What's the current error?"
"Let's simplify this - just make it work, we can style later"
"Revert the last change and try a different approach"
```

---

## Expanding Your Projects

Once you complete a template, try adding:

### Landing Page Extensions
- Contact form that sends email
- Newsletter signup
- Blog section
- Pricing table
- FAQ accordion

### Dashboard Extensions
- Real-time data updates
- Export to CSV
- Date range filters
- Multiple chart types
- Drill-down details

### Web App Extensions
- Team/organization features
- Notifications
- Search functionality
- File uploads
- API integration
