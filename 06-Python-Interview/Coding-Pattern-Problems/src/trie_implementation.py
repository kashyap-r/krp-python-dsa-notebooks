import re

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.frequency = 0  # Stores popularity count for autocomplete ranking

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """Inserts a word into the Trie and increments its frequency count."""
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True
        node.frequency += 1  # Increment popularity every time it's searched/inserted

    def search(self, word):
        """Returns True if the exact word exists in the Trie."""
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    def starts_with(self, prefix):
        """Returns True if any word in the Trie starts with the prefix."""
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True

    def autocomplete(self, prefix):
        """Returns a list of words starting with prefix, sorted by popularity (frequency)."""
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]
            
        results = []
        self._find_all_words(node, prefix, results)
        # Sort by frequency descending, then alphabetically
        results.sort(key=lambda x: (-x[1], x[0]))
        return [word for word, freq in results]

    def _find_all_words(self, current_node, current_word, results):
        if current_node.is_end:
            results.append((current_word, current_node.frequency))
            
        for ch, next_node in current_node.children.items():
            self._find_all_words(next_node, current_word + ch, results)

    def delete(self, word):
        """Safely deletes a word from the Trie. Returns True if deleted."""
        def _delete_helper(node, word, depth):
            if depth == len(word):
                if not node.is_end:
                    return False  # Word doesn't exist
                node.is_end = False  # Unmark the word
                return len(node.children) == 0  # If true, parent can delete this character node
            
            ch = word[depth]
            if ch not in node.children:
                return False
                
            should_delete_child = _delete_helper(node.children[ch], word, depth + 1)
            
            if should_delete_child:
                del node.children[ch]
                # If current node has no other children and isn't another word's end, it can be deleted too
                return len(node.children) == 0 and not node.is_end
                
            return False

        return _delete_helper(self.root, word, 0)

# ==========================================
# TEST HARNESS
# ==========================================
def run_trie_tests():
    print("🚀 Initializing Trie and running test harness...\n")
    trie = Trie()
    
    # 1. Test basic insertion and frequency popularity
    # Let's insert 'car' 3 times, 'care' 1 time, 'cat' 5 times, 'can' 2 times
    words_to_insert = (['car'] * 3) + (['care'] * 1) + (['cat'] * 5) + (['can'] * 2)
    for w in words_to_insert:
        trie.insert(w)
    print("✅ Inserted dataset: 'car' (x3), 'care' (x1), 'cat' (x5), 'can' (x2)")

    # 2. Test exact searches
    print("\n--- 🔍 Testing Exact Search ---")
    print(f"Search 'car':  {trie.search('car')} (Expected: True)")
    print(f"Search 'care': {trie.search('care')} (Expected: True)")
    print(f"Search 'cap':  {trie.search('cap')} (Expected: False)")

    # 3. Test prefixes
    print("\n--- 📂 Testing Starts_With (Prefix Check) ---")
    print(f"Starts with 'ca': {trie.starts_with('ca')} (Expected: True)")
    print(f"Starts with 'ce': {trie.starts_with('ce')} (Expected: False)")

    # 4. Test Autocomplete with Frequency Sorting
    print("\n--- 💡 Testing Autocomplete (Sorted by Frequency) ---")
    # 'cat' (5), 'car' (3), 'can' (2), 'care' (1)
    print(f"Autocomplete for 'ca': {trie.autocomplete('ca')}")
    print(f"Expected Order:        ['cat', 'car', 'can', 'care']")
    print(f"Autocomplete for 'car': {trie.autocomplete('car')}")
    print(f"Expected Order:        ['car', 'care']")

    # 5. Test Deletion
    print("\n--- 🗑️ Testing Word Deletion ---")
    print("Deleting 'car'...")
    trie.delete("car")
    print(f"Search 'car' post-deletion:  {trie.search('car')} (Expected: False)")
    print(f"Search 'care' post-deletion: {trie.search('care')} (Expected: True) <- Ensures shared prefix wasn't broken!")
    print(f"Autocomplete for 'ca' now:   {trie.autocomplete('ca')} (Expected: 'car' is missing)")

if __name__ == "__main__":
    run_trie_tests()
