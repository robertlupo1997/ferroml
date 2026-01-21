---
name: typescript-pro
description: Advanced TypeScript development with full type safety. Use when: (1) designing complex generic types or utility types, (2) achieving end-to-end type safety (tRPC, GraphQL codegen), (3) optimizing TypeScript build performance, (4) migrating JavaScript to TypeScript, (5) authoring type-safe libraries, (6) debugging complex type errors.
---

# TypeScript Pro

## Workflow

1. **Audit** - Review tsconfig strictness, type coverage, build times
2. **Design** - Define domain types first, derive others with mapped/conditional types
3. **Implement** - Use type inference, avoid explicit types where inferred
4. **Validate** - Test types with `expectType`, check edge cases
5. **Optimize** - Profile tsc, use project references for large codebases

## Strict Mode Essentials

```jsonc
// tsconfig.json - non-negotiable settings
{
  "compilerOptions": {
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "exactOptionalPropertyTypes": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true
  }
}
```

## Type Patterns

### Branded Types (nominal typing)

```typescript
type UserId = string & { readonly __brand: 'UserId' };
type OrderId = string & { readonly __brand: 'OrderId' };

// Prevents mixing up IDs even though both are strings
function getUser(id: UserId) { /* ... */ }
```

### Discriminated Unions (state machines)

```typescript
type State =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; data: Data }
  | { status: 'error'; error: Error };

// Exhaustive checking with never
function handle(state: State) {
  switch (state.status) {
    case 'idle': return /* ... */;
    case 'loading': return /* ... */;
    case 'success': return state.data;
    case 'error': return state.error;
    default: return state satisfies never;
  }
}
```

### Const Assertions

```typescript
// Without: routes is string[]
const routes = ['home', 'about', 'contact'];

// With: routes is readonly ['home', 'about', 'contact']
const routes = ['home', 'about', 'contact'] as const;
type Route = typeof routes[number]; // 'home' | 'about' | 'contact'
```

## Build Performance

| Problem | Solution |
|---------|----------|
| Slow full builds | Project references, incremental: true |
| Slow IDE | Exclude node_modules, use skipLibCheck |
| Large bundles | isolatedModules, type-only imports |
| CI bottleneck | Cache .tsbuildinfo, parallel type-check |

## End-to-End Type Safety

```
Database → ORM types → API types → Client types
         Prisma       tRPC/Zod    Inferred
```

- **tRPC**: Types flow from backend to frontend automatically
- **Zod**: Runtime validation + TypeScript types from one source
- **Prisma**: Database schema generates TypeScript types

## Common Type Errors Decoded

| Error | Meaning | Fix |
|-------|---------|-----|
| `Type 'X' is not assignable to type 'Y'` | Shape mismatch | Check property names/types |
| `'X' is possibly 'undefined'` | Nullable access | Add null check or `!` (carefully) |
| `Type instantiation is excessively deep` | Recursive type limit | Simplify or add base case |
| `Cannot find name 'X'` | Missing import/declaration | Import or declare ambient type |

## Library Authoring Checklist

- [ ] `declaration: true` in tsconfig
- [ ] Export types explicitly in package.json `exports`
- [ ] Test types with `@ts-expect-error` comments
- [ ] Support both ESM and CJS if needed
- [ ] Document generic constraints in JSDoc
