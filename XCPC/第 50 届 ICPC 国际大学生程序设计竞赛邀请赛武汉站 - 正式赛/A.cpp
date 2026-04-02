#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define vvi vector<vi>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
// #define endl endl << flush
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
const int MOD = 1e9 + 7;
const int mod = 998244353;
void Murasame()
{
    int n, q;
    cin >> n >> q;
    vi a(n + 1);
    vc<array<int, 2>> need(n + 1);
    ff(i, 1, n)
    {
        cin >> a[i];
        need[i][0] = 1;
        need[i][1] = 1e9;
    }
    while (q--)
    {
        int op, x, y;
        cin >> op >> x >> y;
        need[op][0] = max(need[op][0], x);
        need[op][1] = min(need[op][1], y);
    }
    int ans = 0;
    ff(i, 1, n)
    {
        if (need[i][0] > need[i][1])
        {
            ans = -1;
            break;
        }
        if (a[i] < need[i][0])
        {
            ans += need[i][0] - a[i];
        }
        else if (a[i] > need[i][1])
        {
            ans += a[i] - need[i][1];
        }
    }
    cout << ans << endl;
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
        Murasame();
    }
    return 0;
}