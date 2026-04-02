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
int n, m;
string s;
int nex[2][2] = {{1, 0}, {0, 1}};
void Murasame()
{
    cin >> n >> m >> s;
    s = '~' + s + '~';
    int cx = 0, cy = 0;
    int x = 0, y = 0;
    vi ba(n + m, 0), bb(n + m, 0);
    ff(i, 1, n + m - 1)
    {
        if (s[i] == 'D')
            cx++;
        else if (s[i] == 'R')
            cy++;
        ba[i] = cx, bb[i] = cy;
    }
    int ans = 0;
    ff(i, 1, n + m - 1)
    {
        if (s[i] == 'D')
            x++;
        else if (s[i] == 'R')
            y++;
        int l1 = max(x, i - (m - 1 - cy + y));
        int l2 = i - max(y, i - (n - 1 - cx + x));
        ans += l2 - l1 + 1;
    }
    cout << ans + 1 << endl;
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