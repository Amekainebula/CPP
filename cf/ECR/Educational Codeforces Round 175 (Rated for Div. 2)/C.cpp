#include <bits/stdc++.h>
// Finish Time: 2025/2/28 12:43:12 AC
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define lowbit(x) (x & -x)
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
#define INF 0x7fffffffffffffff
#define endl '\n'
using namespace std;
void solve()
{
    int n, k;
    cin >> n >> k;
    string s;
    cin >> s;
    vector<int> val(n + 1);
    for (int i = 1; i <= n; i++)
        cin >> val[i];
    int l = 0, r = 2e9;
    auto check = [&](int mid) -> bool
    {
        int cnt = 0;
        for (int i = 1; i <= n; i++)
        {
            if (s[i - 1] == 'B' && val[i] > mid)
            {
                cnt++;
                while (i <= n && (s[i - 1] == 'B' || val[i] <= mid))
                    i++;
            }
        }
        return cnt <= k;
    };
    while (l < r)
    {
        int mid = (l + r) / 2;
        if (check(mid))

            r = mid;
        else
            l = mid + 1;
    }
    cout << l << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}