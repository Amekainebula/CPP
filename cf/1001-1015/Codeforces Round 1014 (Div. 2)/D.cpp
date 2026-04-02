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
void solve()
{
    int n;
    string s;
    cin >> n >> s;
    string t = "LIT";
    vector<int> cnt(3, 0);
    vector<int> ans;
    ff(i, 0, n - 1)
        ff(j, 0, 2) if (s[i] == t[j]) cnt[j]++;
    if (cnt[0] == 0 || cnt[1] == 0 || cnt[2] == 0)
    {
        cout << -1 << endl;
        return;
    }
    int need = 0;
    ff(i, 0, 2) need = max(need, cnt[i]);
    
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    //
    cin >> _T;
    while (_T--)
    {
        solve();
    }
    return 0;
}