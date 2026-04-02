#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define vc vector
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
int n, k;
bool check(vector<int> arr, int t)
{
    if (t == 0)
        return arr.size() >= k;
    vector<int> cnt(t, 0);
    for (auto num : arr)
        if (num >= 0 && num < t)
            cnt[num]++;
    for (auto c : cnt)
        if (c < k)
            return false;
    vector<int> cur(t, 0);
    int need = 0;
    int sum = 0;
    for (auto num : arr)
    {
        if (num >= 0 && num < t)
        {
            if (cur[num] == 0)
                need++;
            cur[num]++;
        }
        if (need == t)
        {
            sum++;
            fill(cur.begin(), cur.end(), 0);
            need = 0;
            if (sum >= k)
                return true;
        }
    }
    return sum >= k;
}

int qs(vector<int> arr, int k)
{
    int n = arr.size();
    map<int, int> ty;
    for (auto x : arr)
        ty[x]++;
    int mex = 0;
    for (auto x : ty)
    {
        if (x.first != mex)
            break;
        mex++;
    }
    int l = 0, r = mex;
    int ans = 0;
    while (l <= r)
    {
        int mid = (l + r) / 2;
        if (check(arr, mid))
        {
            ans = mid;
            l = mid + 1;
        }
        else
        {
            r = mid - 1;
        }
    }
    return ans;
}
void solve()
{
    cin >> n >> k;
    vector<int> arr(n);
    for (int i = 0; i < n; i++)
        cin >> arr[i];
    cout << qs(arr, k) << endl;
}
signed main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    int t;
    cin >> t;
    while (t--)
    {
        solve();
    }
    return 0;
}