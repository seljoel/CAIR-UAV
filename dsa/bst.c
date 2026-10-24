/*
 * Binary Search Tree (BST) Implementation in C
 * Operations: Insertion, Deletion, Find Minimum, Find Maximum,
 *             Inorder Traversal, Search
 */

#include <stdio.h>
#include <stdlib.h>

/* ── Node Definition ──────────────────────────────────────────── */
struct Node {
    int data;
    struct Node *left;
    struct Node *right;
};

/* ── Create a new node ────────────────────────────────────────── */
struct Node *createNode(int value) {
    struct Node *newNode = (struct Node *)malloc(sizeof(struct Node));
    if (newNode == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);
    }
    newNode->data = value;
    newNode->left = NULL;
    newNode->right = NULL;
    return newNode;
}

/* ── Insertion ────────────────────────────────────────────────── */
/*
 * Inserts a value into the BST.
 *   - If tree is empty, create a new root node.
 *   - If value < current node, go left.
 *   - If value > current node, go right.
 *   - Duplicates are ignored.
 *
 * Time Complexity:  O(h)  where h = height of the tree
 *   Best/Avg: O(log n)   Worst (skewed): O(n)
 */
struct Node *insert(struct Node *root, int value) {
    if (root == NULL) {
        return createNode(value);
    }

    if (value < root->data) {
        root->left = insert(root->left, value);
    } else if (value > root->data) {
        root->right = insert(root->right, value);
    }
    /* duplicate values are not inserted */

    return root;
}

/* ── Find Minimum ─────────────────────────────────────────────── */
/*
 * The minimum value in a BST is the leftmost node.
 *
 * Time Complexity: O(h)
 */
struct Node *findMin(struct Node *root) {
    if (root == NULL) {
        printf("Tree is empty!\n");
        return NULL;
    }
    while (root->left != NULL) {
        root = root->left;
    }
    return root;
}

/* ── Find Maximum ─────────────────────────────────────────────── */
/*
 * The maximum value in a BST is the rightmost node.
 *
 * Time Complexity: O(h)
 */
struct Node *findMax(struct Node *root) {
    if (root == NULL) {
        printf("Tree is empty!\n");
        return NULL;
    }
    while (root->right != NULL) {
        root = root->right;
    }
    return root;
}

/* ── Search ───────────────────────────────────────────────────── */
/*
 * Searches for a value in the BST.
 * Returns the node if found, NULL otherwise.
 *
 * Time Complexity: O(h)
 */
struct Node *search(struct Node *root, int value) {
    if (root == NULL || root->data == value) {
        return root;
    }

    if (value < root->data) {
        return search(root->left, value);
    }
    return search(root->right, value);
}

/* ── Deletion ─────────────────────────────────────────────────── */
/*
 * Deletes a node with the given value from the BST.
 * Three cases:
 *   Case 1: Node is a leaf           → simply remove it.
 *   Case 2: Node has one child       → replace node with its child.
 *   Case 3: Node has two children    → replace node's data with its
 *           inorder successor (smallest in right subtree), then delete
 *           the inorder successor.
 *
 * Time Complexity: O(h)
 */
struct Node *delete(struct Node *root, int value) {
    if (root == NULL) {
        printf("Value %d not found in the tree.\n", value);
        return root;
    }

    /* Traverse to find the node */
    if (value < root->data) {
        root->left = delete(root->left, value);
    } else if (value > root->data) {
        root->right = delete(root->right, value);
    } else {
        /* Found the node to delete */

        /* Case 1 & 2: Node with zero or one child */
        if (root->left == NULL) {
            struct Node *temp = root->right;
            free(root);
            return temp;
        } else if (root->right == NULL) {
            struct Node *temp = root->left;
            free(root);
            return temp;
        }

        /* Case 3: Node with two children
         * Get inorder successor (smallest in right subtree) */
        struct Node *successor = findMin(root->right);
        root->data = successor->data;
        root->right = delete(root->right, successor->data);
    }

    return root;
}

/* ── Inorder Traversal (sorted output) ────────────────────────── */
/*
 * Prints elements in ascending order (Left → Root → Right).
 * Time Complexity: O(n)
 */
void inorder(struct Node *root) {
    if (root != NULL) {
        inorder(root->left);
        printf("%d ", root->data);
        inorder(root->right);
    }
}

/* ── Preorder Traversal ───────────────────────────────────────── */
/*
 * Prints elements in Root → Left → Right order.
 * Time Complexity: O(n)
 */
void preorder(struct Node *root) {
    if (root != NULL) {
        printf("%d ", root->data);
        preorder(root->left);
        preorder(root->right);
    }
}

/* ── Postorder Traversal ──────────────────────────────────────── */
/*
 * Prints elements in Left → Right → Root order.
 * Time Complexity: O(n)
 */
void postorder(struct Node *root) {
    if (root != NULL) {
        postorder(root->left);
        postorder(root->right);
        printf("%d ", root->data);
    }
}

/* ── Display Menu ─────────────────────────────────────────────── */
void displayMenu() {
    printf("\n=========================================\n");
    printf("       Binary Search Tree Operations\n");
    printf("=========================================\n");
    printf("  1. Insert\n");
    printf("  2. Delete\n");
    printf("  3. Search\n");
    printf("  4. Find Minimum\n");
    printf("  5. Find Maximum\n");
    printf("  6. Inorder Traversal\n");
    printf("  7. Preorder Traversal\n");
    printf("  8. Postorder Traversal\n");
    printf("  9. Exit\n");
    printf("=========================================\n");
    printf("  Enter your choice: ");
}

/* ── Main Driver ──────────────────────────────────────────────── */
int main() {
    struct Node *root = NULL;
    struct Node *result = NULL;
    int choice, value;

    printf("\n*** Binary Search Tree - DSA Implementation ***\n");

    while (1) {
        displayMenu();
        scanf("%d", &choice);

        switch (choice) {
        case 1:
            printf("  Enter value to insert: ");
            scanf("%d", &value);
            root = insert(root, value);
            printf("  -> %d inserted successfully.\n", value);
            printf("  Inorder: ");
            inorder(root);
            printf("\n");
            break;

        case 2:
            printf("  Enter value to delete: ");
            scanf("%d", &value);
            root = delete(root, value);
            printf("  -> Deletion complete.\n");
            printf("  Inorder: ");
            inorder(root);
            printf("\n");
            break;

        case 3:
            printf("  Enter value to search: ");
            scanf("%d", &value);
            result = search(root, value);
            if (result != NULL) {
                printf("  -> %d found in the tree.\n", result->data);
            } else {
                printf("  -> %d not found in the tree.\n", value);
            }
            break;

        case 4:
            result = findMin(root);
            if (result != NULL) {
                printf("  -> Minimum value: %d\n", result->data);
            }
            break;

        case 5:
            result = findMax(root);
            if (result != NULL) {
                printf("  -> Maximum value: %d\n", result->data);
            }
            break;

        case 6:
            printf("  Inorder:   ");
            inorder(root);
            printf("\n");
            break;

        case 7:
            printf("  Preorder:  ");
            preorder(root);
            printf("\n");
            break;

        case 8:
            printf("  Postorder: ");
            postorder(root);
            printf("\n");
            break;

        case 9:
            printf("  Exiting... Goodbye!\n");
            exit(0);

        default:
            printf("  Invalid choice! Please try again.\n");
        }
    }

    return 0;
}
