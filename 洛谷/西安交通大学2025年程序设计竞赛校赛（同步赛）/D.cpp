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
    int n, m;
    cin >> n >> m;
    vc<vc<char>> mp(n + 10, vc<char>(m + 10)), a1(n + 10, vc<char>(m + 10)), a2(n + 10, vc<char>(m + 10));
    ff(i, 1, n) ff(j, 1, m) cin >> mp[i][j];
    ff(i, 1, n) ff(j, 1, m)
    {
        if (mp[i][j] == 'B' && (mp[i + 1][j] == 'B' || mp[i - 1][j] == 'B' || mp[i][j + 1] == 'B' || mp[i][j - 1] == 'B'))
        {
            cout << "No" << endl;
            return;
        }
    }
    a1 = mp;
    a2 = mp;
    ff(i, 1, n) ff(j, 1, m)
    {
        if (a1[i][j] == 'B') // x对称
        {
            a1[n - i + 1][j] = 'B';
        }
        if (a2[i][j] == 'B') // y对称
        {
            a2[i][m - j + 1] = 'B';
        }
    }
    ff(i, 1, n) ff(j, 1, m) cout << a1[i][j] << " \n"[j == m];
    cout << endl;
    ff(i, 1, n) ff(j, 1, m) cout << a2[i][j] << " \n"[j == m];
    bool ok = 1;
    ff(i, 1, n) ff(j, 1, m)
    {
        if (a1[i][j] == 'B' && (a1[i + 1][j] == 'B' || a1[i - 1][j] == 'B' ||
                                a1[i][j + 1] == 'B' || a1[i][j - 1] == 'B'))
        {
            ok = 0;
        }
        if (!ok)
            break;
    }
    if (ok)
    {
        cout << "Yes" << endl;
        return;
    }
    ok = 1;
    ff(i, 1, n) ff(j, 1, m)
    {
        if (a2[i][j] == 'B' && (a2[i + 1][j] == 'B' || a2[i - 1][j] == 'B' ||
                                a2[i][j + 1] == 'B' || a2[i][j - 1] == 'B'))
        {
            ok = 0;
        }
        if (!ok)
            break;
    }
    if (ok)
        cout << "Yes" << endl;
    else
        cout << "No" << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    // cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}