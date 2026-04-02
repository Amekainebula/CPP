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
    string s;
    cin >> s;
    if (s[0] == s[n - 1])
    {
        cout << (s[0] == 'A' ? "Alice" : "Bob") << endl;
        return;
    }
    if (n == 2)
        cout << (s[0] == 'A' ? "Alice" : "Bob") << endl;
    else
    {
        if (s[n - 1] == s[n - 2])
            cout << (s[n - 1] == 'A' ? "Alice" : "Bob") << endl;
        else
        {
            int ok = 1;
            ff(i, 0, n - 2)
            {
                ok &= s[i] == 'A';
            }
            cout << (ok ? "Alice" : "Bob") << endl;
        }
    }
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}