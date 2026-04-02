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
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
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
void Murasame()
{
    int n;
    cin >> n;
    vi a(n + 1, 0), b(n + 1, 0), c(n + 1, 0);
    ff(i, 1, n) cin >> a[i];
    ff(i, 1, n) cin >> b[i], c[i] = a[i] + b[i];
    map<int, int> mp;
    vi pre;
    int now = -1;
    int cnt = 0;
    for (int i = 1; i <= n; i++)
    {
        if (now == -1)
        {
            now = c[i];
            cnt = 1;
        }
        else if (now == c[i])
        {
            cnt++;
        }
        else
        {
            now = c[i];
            pre.pb(cnt);
            cnt = 1;
        }
    }
    if (cnt)
        pre.pb(cnt);
    int ans = -inf;
    int fg = 1;
    for (auto i : pre)
    {
        int temp = 0;
        for (int j = fg; j <= fg + i - 1; j++)
        {
            temp += a[j] * i;
        }
        fg += i;
        ans = max(ans, temp);
    }
    cout << ans << endl;
    // for (auto i : mp)
    //     cout << i.fi << " " << i.se << endl;
    // cout << endl;
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